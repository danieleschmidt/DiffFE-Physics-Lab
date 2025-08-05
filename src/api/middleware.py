"""Middleware for Flask API."""

import time
import logging
from functools import wraps

try:
    from flask import request, g, current_app, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

logger = logging.getLogger(__name__)


def setup_middleware(app):
    """Setup middleware for Flask app."""
    if not HAS_FLASK:
        logger.warning("Flask not available - skipping middleware setup")
        return
    
    # Request timing middleware
    @app.before_request
    def before_request():
        g.start_time = time.time()
        request.environ['REQUEST_TIME'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    
    @app.after_request
    def after_request(response):
        if hasattr(g, 'start_time'):
            response_time = time.time() - g.start_time
            response.headers['X-Response-Time'] = f"{response_time:.3f}s"
            
            # Log slow requests
            if response_time > 1.0:
                logger.warning(f"Slow request: {request.method} {request.path} took {response_time:.3f}s")
        
        return response
    
    # CORS headers (already handled in main app.py but adding here for completeness)
    @app.after_request
    def add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    # Request validation middleware
    @app.before_request
    def validate_request():
        # Skip validation for certain endpoints
        skip_validation = ['/health', '/api/info', '/api/operators']
        if request.path in skip_validation:
            return
        
        # Check content length for POST/PUT requests
        if request.method in ['POST', 'PUT']:
            max_content_length = current_app.config.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)
            if request.content_length and request.content_length > max_content_length:
                return jsonify({
                    'error': f'Request too large. Maximum size: {max_content_length} bytes'
                }), 413
        
        # Validate JSON content type for JSON endpoints
        if (request.method in ['POST', 'PUT'] and 
            request.path.startswith('/api/') and
            request.path not in ['/api/info', '/api/operators']):
            
            if not request.is_json and request.content_length > 0:
                return jsonify({
                    'error': 'Content-Type must be application/json'
                }), 400
    
    # Rate limiting (simple implementation)
    request_counts = {}
    
    @app.before_request
    def rate_limit():
        # Skip rate limiting for health checks
        if request.path in ['/health', '/api/info']:
            return
        
        client_ip = request.environ.get('REMOTE_ADDR', 'unknown')
        current_time = time.time()
        
        # Clean old entries (older than 1 minute)
        request_counts[client_ip] = [
            req_time for req_time in request_counts.get(client_ip, [])
            if current_time - req_time < 60
        ]
        
        # Check rate limit (100 requests per minute per IP)
        if len(request_counts.get(client_ip, [])) > 100:
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': 'Too many requests. Limit: 100 requests per minute.'
            }), 429
        
        # Add current request
        if client_ip not in request_counts:
            request_counts[client_ip] = []
        request_counts[client_ip].append(current_time)
    
    # Error logging middleware
    @app.teardown_request
    def log_request_info(exception):
        if exception:
            logger.error(f"Request failed: {request.method} {request.path} - {exception}")
    
    logger.info("Middleware setup completed")


def require_auth(f):
    """Decorator for endpoints requiring authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Simple API key authentication
        api_key = request.headers.get('Authorization')
        
        if not api_key:
            return jsonify({'error': 'Missing Authorization header'}), 401
        
        # Remove 'Bearer ' prefix if present
        if api_key.startswith('Bearer '):
            api_key = api_key[7:]
        
        # In production, validate against database/service
        valid_keys = current_app.config.get('API_KEYS', [])
        
        if api_key not in valid_keys:
            return jsonify({'error': 'Invalid API key'}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function


def validate_content_type(content_type='application/json'):
    """Decorator to validate request content type."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.content_type != content_type:
                return jsonify({
                    'error': f'Content-Type must be {content_type}'
                }), 400
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def validate_json_schema(schema):
    """Decorator to validate JSON request against schema."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({'error': 'Request must be JSON'}), 400
            
            try:
                data = request.get_json()
                
                # Basic schema validation
                if 'required' in schema:
                    for field in schema['required']:
                        if field not in data:
                            return jsonify({
                                'error': f'Missing required field: {field}'
                            }), 400
                
                if 'properties' in schema:
                    for field, field_schema in schema['properties'].items():
                        if field in data:
                            field_type = field_schema.get('type')
                            if field_type and not isinstance(data[field], 
                                                           {'string': str, 'number': (int, float), 
                                                            'boolean': bool, 'array': list, 
                                                            'object': dict}.get(field_type, type(None))):
                                return jsonify({
                                    'error': f'Field {field} must be of type {field_type}'
                                }), 400
                
            except Exception as e:
                return jsonify({'error': f'JSON validation error: {str(e)}'}), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def log_performance(f):
    """Decorator to log performance metrics."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = f(*args, **kwargs)
            end_time = time.time()
            
            logger.info(f"Performance: {f.__name__} took {end_time - start_time:.3f}s")
            
            return result
        
        except Exception as e:
            end_time = time.time()
            logger.error(f"Performance: {f.__name__} failed after {end_time - start_time:.3f}s: {e}")
            raise
    
    return decorated_function


def cache_response(timeout=300):
    """Decorator to cache responses."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Simple in-memory cache (in production, use Redis or similar)
            if not hasattr(current_app, '_response_cache'):
                current_app._response_cache = {}
            
            # Create cache key
            cache_key = f"{request.method}:{request.path}:{request.query_string.decode()}"
            current_time = time.time()
            
            # Check cache
            if cache_key in current_app._response_cache:
                cached_response, cached_time = current_app._response_cache[cache_key]
                if current_time - cached_time < timeout:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached_response
            
            # Call function and cache result
            result = f(*args, **kwargs)
            current_app._response_cache[cache_key] = (result, current_time)
            
            logger.debug(f"Cached response for {cache_key}")
            
            return result
        
        return decorated_function
    return decorator


def sanitize_input(f):
    """Decorator to sanitize input data."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.is_json:
            data = request.get_json()
            
            # Basic sanitization
            def sanitize_dict(d):
                if isinstance(d, dict):
                    return {k: sanitize_dict(v) for k, v in d.items() 
                           if not str(k).startswith('_')}  # Remove private fields
                elif isinstance(d, list):
                    return [sanitize_dict(item) for item in d]
                elif isinstance(d, str):
                    # Basic string sanitization
                    return d.strip()[:1000]  # Limit string length
                else:
                    return d
            
            # Replace request data with sanitized version
            request._cached_json = sanitize_dict(data)
        
        return f(*args, **kwargs)
    
    return decorated_function