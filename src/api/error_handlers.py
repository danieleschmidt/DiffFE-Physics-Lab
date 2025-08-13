"""Error handlers for Flask API."""

import logging
import traceback
from typing import Any, Dict, Tuple

try:
    from flask import current_app, jsonify, request

    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

logger = logging.getLogger(__name__)


def register_error_handlers(app):
    """Register error handlers with Flask app."""
    if not HAS_FLASK:
        logger.warning("Flask not available - skipping error handler registration")
        return

    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request errors."""
        logger.warning(f"Bad request: {request.method} {request.path} - {error}")
        return (
            jsonify(
                {
                    "error": "Bad Request",
                    "message": "The request could not be understood by the server",
                    "status_code": 400,
                }
            ),
            400,
        )

    @app.errorhandler(401)
    def unauthorized(error):
        """Handle 401 Unauthorized errors."""
        logger.warning(f"Unauthorized access: {request.method} {request.path}")
        return (
            jsonify(
                {
                    "error": "Unauthorized",
                    "message": "Authentication is required to access this resource",
                    "status_code": 401,
                }
            ),
            401,
        )

    @app.errorhandler(403)
    def forbidden(error):
        """Handle 403 Forbidden errors."""
        logger.warning(f"Forbidden access: {request.method} {request.path}")
        return (
            jsonify(
                {
                    "error": "Forbidden",
                    "message": "You do not have permission to access this resource",
                    "status_code": 403,
                }
            ),
            403,
        )

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors."""
        logger.info(f"Resource not found: {request.method} {request.path}")
        return (
            jsonify(
                {
                    "error": "Not Found",
                    "message": "The requested resource could not be found",
                    "status_code": 404,
                }
            ),
            404,
        )

    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 Method Not Allowed errors."""
        logger.warning(f"Method not allowed: {request.method} {request.path}")
        return (
            jsonify(
                {
                    "error": "Method Not Allowed",
                    "message": f"The {request.method} method is not allowed for this endpoint",
                    "status_code": 405,
                    "allowed_methods": (
                        error.description.get("valid_methods", [])
                        if hasattr(error, "description")
                        else []
                    ),
                }
            ),
            405,
        )

    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle 413 Request Entity Too Large errors."""
        max_size = current_app.config.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024)
        logger.warning(f"Request too large: {request.method} {request.path}")
        return (
            jsonify(
                {
                    "error": "Request Too Large",
                    "message": f"Request size exceeds maximum allowed size of {max_size} bytes",
                    "status_code": 413,
                    "max_size_bytes": max_size,
                }
            ),
            413,
        )

    @app.errorhandler(415)
    def unsupported_media_type(error):
        """Handle 415 Unsupported Media Type errors."""
        logger.warning(
            f"Unsupported media type: {request.method} {request.path} - {request.content_type}"
        )
        return (
            jsonify(
                {
                    "error": "Unsupported Media Type",
                    "message": "The media type of the request is not supported",
                    "status_code": 415,
                    "content_type": request.content_type,
                    "supported_types": ["application/json"],
                }
            ),
            415,
        )

    @app.errorhandler(422)
    def unprocessable_entity(error):
        """Handle 422 Unprocessable Entity errors."""
        logger.warning(f"Unprocessable entity: {request.method} {request.path}")

        # Extract validation errors if available
        validation_errors = []
        if hasattr(error, "data"):
            validation_errors = error.data.get("messages", [])

        return (
            jsonify(
                {
                    "error": "Unprocessable Entity",
                    "message": "The request was well-formed but contains semantic errors",
                    "status_code": 422,
                    "validation_errors": validation_errors,
                }
            ),
            422,
        )

    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        """Handle 429 Too Many Requests errors."""
        logger.warning(f"Rate limit exceeded: {request.method} {request.path}")
        return (
            jsonify(
                {
                    "error": "Too Many Requests",
                    "message": "Rate limit exceeded. Please try again later.",
                    "status_code": 429,
                    "retry_after": 60,  # seconds
                }
            ),
            429,
        )

    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle 500 Internal Server Error."""
        logger.error(f"Internal server error: {request.method} {request.path}")
        logger.error(f"Error details: {error}")

        # Log full traceback in debug mode
        if current_app.debug:
            logger.error(traceback.format_exc())

        return (
            jsonify(
                {
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred on the server",
                    "status_code": 500,
                }
            ),
            500,
        )

    @app.errorhandler(501)
    def not_implemented(error):
        """Handle 501 Not Implemented errors."""
        logger.warning(f"Not implemented: {request.method} {request.path}")
        return (
            jsonify(
                {
                    "error": "Not Implemented",
                    "message": "This feature is not yet implemented",
                    "status_code": 501,
                }
            ),
            501,
        )

    @app.errorhandler(502)
    def bad_gateway(error):
        """Handle 502 Bad Gateway errors."""
        logger.error(f"Bad gateway: {request.method} {request.path}")
        return (
            jsonify(
                {
                    "error": "Bad Gateway",
                    "message": "The server received an invalid response from an upstream server",
                    "status_code": 502,
                }
            ),
            502,
        )

    @app.errorhandler(503)
    def service_unavailable(error):
        """Handle 503 Service Unavailable errors."""
        logger.error(f"Service unavailable: {request.method} {request.path}")
        return (
            jsonify(
                {
                    "error": "Service Unavailable",
                    "message": "The service is temporarily unavailable. Please try again later.",
                    "status_code": 503,
                }
            ),
            503,
        )

    @app.errorhandler(504)
    def gateway_timeout(error):
        """Handle 504 Gateway Timeout errors."""
        logger.error(f"Gateway timeout: {request.method} {request.path}")
        return (
            jsonify(
                {
                    "error": "Gateway Timeout",
                    "message": "The server did not receive a timely response from an upstream server",
                    "status_code": 504,
                }
            ),
            504,
        )

    # Handle specific exceptions
    @app.errorhandler(ValueError)
    def handle_value_error(error):
        """Handle ValueError exceptions."""
        logger.warning(f"Value error: {request.method} {request.path} - {error}")
        return (
            jsonify(
                {"error": "Validation Error", "message": str(error), "status_code": 400}
            ),
            400,
        )

    @app.errorhandler(TypeError)
    def handle_type_error(error):
        """Handle TypeError exceptions."""
        logger.error(f"Type error: {request.method} {request.path} - {error}")

        if current_app.debug:
            return (
                jsonify(
                    {
                        "error": "Type Error",
                        "message": str(error),
                        "status_code": 500,
                        "traceback": traceback.format_exc(),
                    }
                ),
                500,
            )
        else:
            return (
                jsonify(
                    {
                        "error": "Internal Server Error",
                        "message": "A type error occurred",
                        "status_code": 500,
                    }
                ),
                500,
            )

    @app.errorhandler(ImportError)
    def handle_import_error(error):
        """Handle ImportError exceptions."""
        logger.error(f"Import error: {request.method} {request.path} - {error}")
        return (
            jsonify(
                {
                    "error": "Dependency Error",
                    "message": f"Required dependency not available: {str(error)}",
                    "status_code": 500,
                }
            ),
            500,
        )

    @app.errorhandler(FileNotFoundError)
    def handle_file_not_found_error(error):
        """Handle FileNotFoundError exceptions."""
        logger.error(f"File not found: {request.method} {request.path} - {error}")
        return (
            jsonify(
                {
                    "error": "File Not Found",
                    "message": "A required file could not be found",
                    "status_code": 500,
                }
            ),
            500,
        )

    @app.errorhandler(PermissionError)
    def handle_permission_error(error):
        """Handle PermissionError exceptions."""
        logger.error(f"Permission error: {request.method} {request.path} - {error}")
        return (
            jsonify(
                {
                    "error": "Permission Error",
                    "message": "Insufficient permissions to access required resources",
                    "status_code": 500,
                }
            ),
            500,
        )

    @app.errorhandler(TimeoutError)
    def handle_timeout_error(error):
        """Handle TimeoutError exceptions."""
        logger.error(f"Timeout error: {request.method} {request.path} - {error}")
        return (
            jsonify(
                {
                    "error": "Timeout Error",
                    "message": "The operation timed out",
                    "status_code": 504,
                }
            ),
            504,
        )

    @app.errorhandler(MemoryError)
    def handle_memory_error(error):
        """Handle MemoryError exceptions."""
        logger.critical(f"Memory error: {request.method} {request.path} - {error}")
        return (
            jsonify(
                {
                    "error": "Memory Error",
                    "message": "Insufficient memory to complete the operation",
                    "status_code": 507,  # Insufficient Storage
                    "suggestion": "Try reducing the problem size or contact support",
                }
            ),
            507,
        )

    # Catch-all for unhandled exceptions
    @app.errorhandler(Exception)
    def handle_generic_exception(error):
        """Handle any unhandled exceptions."""
        logger.error(
            f"Unhandled exception: {request.method} {request.path} - {type(error).__name__}: {error}"
        )

        if current_app.debug:
            logger.error(traceback.format_exc())

            return (
                jsonify(
                    {
                        "error": f"{type(error).__name__}",
                        "message": str(error),
                        "status_code": 500,
                        "traceback": traceback.format_exc(),
                    }
                ),
                500,
            )
        else:
            return (
                jsonify(
                    {
                        "error": "Internal Server Error",
                        "message": "An unexpected error occurred",
                        "status_code": 500,
                    }
                ),
                500,
            )

    logger.info("Error handlers registered")


class APIException(Exception):
    """Base class for API exceptions."""

    def __init__(
        self, message: str, status_code: int = 500, details: Dict[str, Any] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        result = {
            "error": self.__class__.__name__,
            "message": self.message,
            "status_code": self.status_code,
        }

        if self.details:
            result.update(self.details)

        return result


class ValidationError(APIException):
    """Exception for validation errors."""

    def __init__(self, message: str, field: str = None, details: Dict[str, Any] = None):
        super().__init__(message, 400, details)
        self.field = field

        if field:
            self.details["field"] = field


class AuthenticationError(APIException):
    """Exception for authentication errors."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, 401)


class AuthorizationError(APIException):
    """Exception for authorization errors."""

    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, 403)


class NotFoundError(APIException):
    """Exception for resource not found errors."""

    def __init__(
        self, message: str, resource_type: str = None, resource_id: str = None
    ):
        super().__init__(message, 404)

        if resource_type:
            self.details["resource_type"] = resource_type
        if resource_id:
            self.details["resource_id"] = resource_id


class ConflictError(APIException):
    """Exception for resource conflict errors."""

    def __init__(self, message: str, resource_type: str = None):
        super().__init__(message, 409)

        if resource_type:
            self.details["resource_type"] = resource_type


class RateLimitError(APIException):
    """Exception for rate limit errors."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        super().__init__(message, 429)
        self.details["retry_after"] = retry_after


class DependencyError(APIException):
    """Exception for missing dependency errors."""

    def __init__(self, message: str, dependency: str = None):
        super().__init__(message, 500)

        if dependency:
            self.details["missing_dependency"] = dependency


def create_error_response(
    error: str, message: str, status_code: int = 500, details: Dict[str, Any] = None
) -> Tuple[Dict[str, Any], int]:
    """Create standardized error response.

    Parameters
    ----------
    error : str
        Error type/name
    message : str
        Error message
    status_code : int, optional
        HTTP status code, by default 500
    details : Dict[str, Any], optional
        Additional error details

    Returns
    -------
    Tuple[Dict[str, Any], int]
        Error response dictionary and status code
    """
    response = {"error": error, "message": message, "status_code": status_code}

    if details:
        response.update(details)

    return response, status_code
