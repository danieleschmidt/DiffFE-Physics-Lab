"""API route definitions for DiffFE-Physics-Lab."""

import json
import logging
from typing import Dict, Any, Optional
from functools import wraps

try:
    from flask import Blueprint, request, jsonify, current_app
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

from ..models import Problem, FEBMLProblem
from ..services import FEBMLSolver, OptimizationService, AssemblyEngine
from ..utils import validate_mesh, compute_error
from ..utils.manufactured_solutions import generate_manufactured_solution, SolutionType

logger = logging.getLogger(__name__)

if HAS_FLASK:
    api_bp = Blueprint('api', __name__, url_prefix='/api')
else:
    api_bp = None


def require_json(f):
    """Decorator to require JSON content type."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        return f(*args, **kwargs)
    return decorated_function


def handle_errors(f):
    """Decorator to handle common API errors."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error: {e}")
            return jsonify({'error': f'Validation error: {str(e)}'}), 400
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return jsonify({'error': f'Missing dependency: {str(e)}'}), 500
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    return decorated_function


if HAS_FLASK:
    
    @api_bp.route('/problems', methods=['POST'])
    @require_json
    @handle_errors
    def create_problem():
        """Create a new FEM problem."""
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['mesh_config', 'function_space_config']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        try:
            # This would create actual mesh/function space with Firedrake
            # For now, return success with metadata
            problem_id = f"problem_{len(current_app.problems) + 1}"
            
            problem_info = {
                'id': problem_id,
                'mesh_config': data['mesh_config'],
                'function_space_config': data['function_space_config'],
                'equations': [],
                'boundary_conditions': [],
                'status': 'created'
            }
            
            # Store problem (in production, would use database)
            if not hasattr(current_app, 'problems'):
                current_app.problems = {}
            current_app.problems[problem_id] = problem_info
            
            logger.info(f"Created problem {problem_id}")
            
            return jsonify({
                'success': True,
                'problem_id': problem_id,
                'message': 'Problem created successfully'
            }), 201
            
        except Exception as e:
            logger.error(f"Failed to create problem: {e}")
            return jsonify({'error': str(e)}), 500
    
    
    @api_bp.route('/problems/<problem_id>', methods=['GET'])
    @handle_errors
    def get_problem(problem_id: str):
        """Get problem information."""
        if not hasattr(current_app, 'problems'):
            return jsonify({'error': 'No problems found'}), 404
        
        if problem_id not in current_app.problems:
            return jsonify({'error': f'Problem {problem_id} not found'}), 404
        
        problem_info = current_app.problems[problem_id]
        return jsonify(problem_info)
    
    
    @api_bp.route('/problems/<problem_id>/equations', methods=['POST'])
    @require_json
    @handle_errors
    def add_equation(problem_id: str):
        """Add equation to problem."""
        if not hasattr(current_app, 'problems') or problem_id not in current_app.problems:
            return jsonify({'error': f'Problem {problem_id} not found'}), 404
        
        data = request.get_json()
        
        # Validate equation data
        required_fields = ['type', 'parameters']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        equation_info = {
            'type': data['type'],
            'parameters': data['parameters'],
            'name': data.get('name', f"equation_{len(current_app.problems[problem_id]['equations']) + 1}")
        }
        
        current_app.problems[problem_id]['equations'].append(equation_info)
        
        logger.info(f"Added equation to problem {problem_id}")
        
        return jsonify({
            'success': True,
            'equation': equation_info,
            'message': 'Equation added successfully'
        })
    
    
    @api_bp.route('/problems/<problem_id>/solve', methods=['POST'])
    @require_json
    @handle_errors
    def solve_problem(problem_id: str):
        """Solve FEM problem."""
        if not hasattr(current_app, 'problems') or problem_id not in current_app.problems:
            return jsonify({'error': f'Problem {problem_id} not found'}), 404
        
        data = request.get_json()
        problem_info = current_app.problems[problem_id]
        
        # Solver options
        solver_options = data.get('solver_options', {})
        backend = solver_options.get('backend', 'jax')
        
        try:
            # In production, would create actual Firedrake problem and solve
            # For now, simulate successful solve
            solution_id = f"solution_{problem_id}_{len(current_app.solutions) + 1 if hasattr(current_app, 'solutions') else 1}"
            
            solution_info = {
                'id': solution_id,
                'problem_id': problem_id,
                'solver_options': solver_options,
                'backend': backend,
                'status': 'solved',
                'metadata': {
                    'dofs': 1000,  # Placeholder
                    'solve_time': 0.123,  # Placeholder
                    'iterations': 10
                }
            }
            
            # Store solution
            if not hasattr(current_app, 'solutions'):
                current_app.solutions = {}
            current_app.solutions[solution_id] = solution_info
            
            # Update problem status
            current_app.problems[problem_id]['status'] = 'solved'
            current_app.problems[problem_id]['latest_solution'] = solution_id
            
            logger.info(f"Solved problem {problem_id}, solution {solution_id}")
            
            return jsonify({
                'success': True,
                'solution_id': solution_id,
                'metadata': solution_info['metadata'],
                'message': 'Problem solved successfully'
            })
            
        except Exception as e:
            logger.error(f"Failed to solve problem {problem_id}: {e}")
            return jsonify({'error': str(e)}), 500
    
    
    @api_bp.route('/solutions/<solution_id>', methods=['GET'])
    @handle_errors
    def get_solution(solution_id: str):
        """Get solution information."""
        if not hasattr(current_app, 'solutions'):
            return jsonify({'error': 'No solutions found'}), 404
        
        if solution_id not in current_app.solutions:
            return jsonify({'error': f'Solution {solution_id} not found'}), 404
        
        solution_info = current_app.solutions[solution_id]
        return jsonify(solution_info)
    
    
    @api_bp.route('/optimization', methods=['POST'])
    @require_json
    @handle_errors
    def run_optimization():
        """Run parameter optimization."""
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['problem_id', 'objective', 'parameters']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        problem_id = data['problem_id']
        objective_config = data['objective']
        optimization_params = data['parameters']
        
        try:
            # Simulate optimization run
            optimization_id = f"opt_{problem_id}_{len(current_app.optimizations) + 1 if hasattr(current_app, 'optimizations') else 1}"
            
            optimization_result = {
                'id': optimization_id,
                'problem_id': problem_id,
                'objective_config': objective_config,
                'parameters': optimization_params,
                'status': 'completed',
                'result': {
                    'success': True,
                    'optimal_parameters': {'param1': 1.23, 'param2': 4.56},  # Placeholder
                    'optimal_value': 0.001234,
                    'iterations': 25,
                    'function_evaluations': 48,
                    'message': 'Optimization converged'
                }
            }
            
            # Store optimization
            if not hasattr(current_app, 'optimizations'):
                current_app.optimizations = {}
            current_app.optimizations[optimization_id] = optimization_result
            
            logger.info(f"Completed optimization {optimization_id}")
            
            return jsonify({
                'success': True,
                'optimization_id': optimization_id,
                'result': optimization_result['result']
            })
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    
    @api_bp.route('/meshes', methods=['POST'])
    @require_json
    @handle_errors
    def create_mesh():
        """Create computational mesh."""
        data = request.get_json()
        
        # Validate mesh configuration
        required_fields = ['type', 'parameters']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        mesh_type = data['type']
        mesh_params = data['parameters']
        
        # Supported mesh types
        supported_types = ['unit_square', 'unit_cube', 'unit_interval', 'custom']
        if mesh_type not in supported_types:
            return jsonify({
                'error': f'Unsupported mesh type: {mesh_type}. Supported: {supported_types}'
            }), 400
        
        try:
            # Simulate mesh creation
            mesh_id = f"mesh_{len(current_app.meshes) + 1 if hasattr(current_app, 'meshes') else 1}"
            
            mesh_info = {
                'id': mesh_id,
                'type': mesh_type,
                'parameters': mesh_params,
                'metadata': {
                    'num_cells': 1000,  # Placeholder
                    'num_vertices': 521,  # Placeholder
                    'dimension': mesh_params.get('dimension', 2)
                },
                'status': 'created'
            }
            
            # Store mesh
            if not hasattr(current_app, 'meshes'):
                current_app.meshes = {}
            current_app.meshes[mesh_id] = mesh_info
            
            logger.info(f"Created mesh {mesh_id}")
            
            return jsonify({
                'success': True,
                'mesh_id': mesh_id,
                'metadata': mesh_info['metadata']
            }), 201
            
        except Exception as e:
            logger.error(f"Failed to create mesh: {e}")
            return jsonify({'error': str(e)}), 500
    
    
    @api_bp.route('/operators', methods=['GET'])
    @handle_errors
    def list_operators():
        """List available operators."""
        operators = [
            {
                'name': 'laplacian',
                'type': 'linear',
                'description': 'Laplacian operator for diffusion problems',
                'parameters': ['diffusion_coeff', 'source']
            },
            {
                'name': 'elasticity',
                'type': 'linear',
                'description': 'Linear elasticity operator',
                'parameters': ['youngs_modulus', 'poissons_ratio', 'body_force']
            },
            {
                'name': 'navier_stokes',
                'type': 'nonlinear', 
                'description': 'Navier-Stokes operator for fluid flow',
                'parameters': ['reynolds_number', 'body_force']
            },
            {
                'name': 'physics_informed_sentiment',
                'type': 'ml',
                'description': 'Physics-informed sentiment classifier using energy conservation',
                'parameters': ['vocab_size', 'embedding_dim', 'physics_weight']
            },
            {
                'name': 'diffusion_sentiment',
                'type': 'ml',
                'description': 'Sentiment propagation using diffusion equations',
                'parameters': ['diffusion_rate', 'time_steps']
            },
            {
                'name': 'conservation_sentiment',
                'type': 'ml',
                'description': 'Sentiment analyzer with conservation principles',
                'parameters': ['conservation_weight']
            }
        ]
        
        return jsonify({
            'operators': operators,
            'count': len(operators)
        })
    
    
    @api_bp.route('/operators/<operator_name>/mms', methods=['GET'])
    @handle_errors
    def get_manufactured_solution(operator_name: str):
        """Get manufactured solution for operator."""
        
        # Validate operator
        supported_operators = ['laplacian', 'elasticity']
        if operator_name not in supported_operators:
            return jsonify({
                'error': f'MMS not available for operator: {operator_name}'
            }), 400
        
        # Parse query parameters
        dimension = request.args.get('dimension', 2, type=int)
        solution_type = request.args.get('type', 'trigonometric')
        frequency = request.args.get('frequency', 1.0, type=float)
        
        try:
            # Generate manufactured solution
            sol_type = SolutionType(solution_type)
            mms = generate_manufactured_solution(
                solution_type=sol_type,
                dimension=dimension,
                parameters={'frequency': frequency}
            )
            
            # Convert functions to serializable format
            test_points = [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]
            
            mms_data = {
                'type': mms['type'],
                'dimension': dimension,
                'parameters': {'frequency': frequency},
                'sample_evaluations': []
            }
            
            for point in test_points:
                mms_data['sample_evaluations'].append({
                    'point': point,
                    'solution': float(mms['solution'](point)),
                    'source': float(mms['source'](point))
                })
            
            return jsonify({
                'success': True,
                'manufactured_solution': mms_data
            })
            
        except Exception as e:
            logger.error(f"Failed to generate MMS for {operator_name}: {e}")
            return jsonify({'error': str(e)}), 500
    
    
    @api_bp.route('/validation/convergence', methods=['POST'])
    @require_json
    @handle_errors
    def validate_convergence():
        """Validate convergence rates."""
        data = request.get_json()
        
        required_fields = ['mesh_sizes', 'errors']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        mesh_sizes = data['mesh_sizes']
        errors = data['errors']
        
        if len(mesh_sizes) != len(errors) or len(mesh_sizes) < 2:
            return jsonify({
                'error': 'Need at least 2 matching mesh sizes and errors'
            }), 400
        
        try:
            from ..utils.error_computation import compute_convergence_rate
            
            convergence_result = compute_convergence_rate(mesh_sizes, errors)
            
            return jsonify({
                'success': True,
                'convergence_analysis': convergence_result
            })
            
        except Exception as e:
            logger.error(f"Convergence validation failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    
    # Sentiment Analysis Endpoints
    @api_bp.route('/sentiment/analyze', methods=['POST'])
    @require_json
    @handle_errors
    def analyze_sentiment():
        """Analyze sentiment using physics-informed models."""
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['text']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        text = data['text']
        model_type = data.get('model_type', 'physics_informed')
        model_params = data.get('parameters', {})
        
        try:
            from ..operators.sentiment import create_sentiment_operator
            
            # Create sentiment operator
            operator = create_sentiment_operator(model_type, **model_params)
            
            # Simple tokenization (in production would use proper tokenizer)
            tokens = text.lower().split()
            token_ids = [hash(token) % operator.vocab_size for token in tokens]
            
            if operator.backend == "jax":
                import jax.numpy as jnp
                token_array = jnp.array(token_ids[:100])  # Limit sequence length
            else:
                import torch
                token_array = torch.tensor(token_ids[:100], dtype=torch.long)
            
            # Get sentiment prediction
            prediction = operator.forward(token_array)
            
            # Convert to standard format
            if operator.backend == "jax":
                sentiment_scores = prediction.tolist()
            else:
                sentiment_scores = prediction.detach().numpy().tolist()
            
            # Map to sentiment labels
            sentiment_labels = ['negative', 'neutral', 'positive']
            predicted_class = sentiment_scores.index(max(sentiment_scores))
            
            result = {
                'text': text,
                'model_type': model_type,
                'backend': operator.backend,
                'sentiment': {
                    'predicted_class': sentiment_labels[predicted_class],
                    'confidence': max(sentiment_scores),
                    'scores': {
                        label: score for label, score in zip(sentiment_labels, sentiment_scores)
                    }
                },
                'tokens_processed': len(token_ids),
                'physics_regularization': model_type in ['physics_informed', 'conservation', 'diffusion']
            }
            
            logger.info(f"Analyzed sentiment for text length {len(text)}")
            
            return jsonify({
                'success': True,
                'result': result
            })
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    
    @api_bp.route('/sentiment/batch', methods=['POST'])
    @require_json
    @handle_errors
    def analyze_sentiment_batch():
        """Analyze sentiment for multiple texts."""
        data = request.get_json()
        
        # Validate required fields
        if 'texts' not in data or not isinstance(data['texts'], list):
            return jsonify({'error': 'Field "texts" must be a list of strings'}), 400
        
        texts = data['texts'][:100]  # Limit batch size
        model_type = data.get('model_type', 'physics_informed')
        model_params = data.get('parameters', {})
        
        try:
            from ..operators.sentiment import create_sentiment_operator
            
            # Create sentiment operator
            operator = create_sentiment_operator(model_type, **model_params)
            
            results = []
            sentiment_labels = ['negative', 'neutral', 'positive']
            
            for i, text in enumerate(texts):
                # Simple tokenization
                tokens = text.lower().split()
                token_ids = [hash(token) % operator.vocab_size for token in tokens]
                
                if operator.backend == "jax":
                    import jax.numpy as jnp
                    token_array = jnp.array(token_ids[:100])
                else:
                    import torch
                    token_array = torch.tensor(token_ids[:100], dtype=torch.long)
                
                # Get prediction
                prediction = operator.forward(token_array)
                
                # Convert to standard format
                if operator.backend == "jax":
                    sentiment_scores = prediction.tolist()
                else:
                    sentiment_scores = prediction.detach().numpy().tolist()
                
                predicted_class = sentiment_scores.index(max(sentiment_scores))
                
                results.append({
                    'index': i,
                    'text': text,
                    'sentiment': {
                        'predicted_class': sentiment_labels[predicted_class],
                        'confidence': max(sentiment_scores),
                        'scores': {
                            label: score for label, score in zip(sentiment_labels, sentiment_scores)
                        }
                    }
                })
            
            logger.info(f"Processed batch of {len(texts)} texts")
            
            return jsonify({
                'success': True,
                'results': results,
                'model_type': model_type,
                'backend': operator.backend,
                'batch_size': len(texts)
            })
            
        except Exception as e:
            logger.error(f"Batch sentiment analysis failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    
    @api_bp.route('/sentiment/models', methods=['GET'])
    @handle_errors
    def list_sentiment_models():
        """List available sentiment analysis models."""
        models = [
            {
                'name': 'physics_informed',
                'description': 'Physics-informed sentiment classifier with energy conservation',
                'parameters': {
                    'vocab_size': {'type': 'int', 'default': 10000},
                    'embedding_dim': {'type': 'int', 'default': 128},
                    'hidden_dim': {'type': 'int', 'default': 256},
                    'physics_weight': {'type': 'float', 'default': 0.1}
                },
                'features': ['energy_conservation', 'gradient_flow', 'stable_representations']
            },
            {
                'name': 'diffusion',
                'description': 'Sentiment propagation using heat equation dynamics',
                'parameters': {
                    'diffusion_rate': {'type': 'float', 'default': 0.1},
                    'time_steps': {'type': 'int', 'default': 10}
                },
                'features': ['contextual_propagation', 'temporal_dynamics', 'smooth_transitions']
            },
            {
                'name': 'conservation',
                'description': 'Conservation-based sentiment analysis',
                'parameters': {
                    'conservation_weight': {'type': 'float', 'default': 0.05}
                },
                'features': ['conservation_laws', 'semantic_consistency', 'balanced_predictions']
            }
        ]
        
        return jsonify({
            'models': models,
            'count': len(models),
            'backends': ['jax', 'torch']
        })
    
    
    @api_bp.route('/sentiment/diffuse', methods=['POST'])
    @require_json  
    @handle_errors
    def diffuse_sentiment():
        """Apply sentiment diffusion to text segments."""
        data = request.get_json()
        
        required_fields = ['segments', 'adjacency']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        segments = data['segments']
        adjacency = data['adjacency']
        diffusion_params = data.get('parameters', {})
        
        try:
            from ..operators.sentiment import DiffusionSentimentPropagator
            import numpy as np
            
            # Create diffusion operator
            propagator = DiffusionSentimentPropagator(**diffusion_params)
            
            # Convert to numpy arrays
            initial_sentiment = np.array(segments, dtype=float)
            adjacency_matrix = np.array(adjacency, dtype=float)
            
            # Apply sentiment diffusion
            if propagator.backend == "jax":
                import jax.numpy as jnp
                initial_jnp = jnp.array(initial_sentiment)
                adjacency_jnp = jnp.array(adjacency_matrix)
                
                propagated = propagator.propagate_sentiment(initial_jnp, adjacency_jnp)
                result_sentiment = propagated.tolist()
            else:
                import torch
                initial_torch = torch.tensor(initial_sentiment, dtype=torch.float32)
                adjacency_torch = torch.tensor(adjacency_matrix, dtype=torch.float32)
                
                propagated = propagator.propagate_sentiment(initial_torch, adjacency_torch)
                result_sentiment = propagated.detach().numpy().tolist()
            
            return jsonify({
                'success': True,
                'initial_sentiment': initial_sentiment.tolist(),
                'propagated_sentiment': result_sentiment,
                'diffusion_parameters': diffusion_params,
                'backend': propagator.backend
            })
            
        except Exception as e:
            logger.error(f"Sentiment diffusion failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    
    @api_bp.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0-dev',
            'components': {
                'flask': HAS_FLASK,
                'firedrake': False,  # Would check actual availability
                'jax': True,  # Would check actual availability
                'torch': False  # Would check actual availability
            }
        })


def register_routes(app):
    """Register API routes with Flask app."""
    if not HAS_FLASK:
        logger.warning("Flask not available - skipping route registration")
        return
    
    app.register_blueprint(api_bp)
    logger.info("API routes registered")


# Error handlers for the blueprint
if HAS_FLASK:
    
    @api_bp.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Resource not found'}), 404
    
    
    @api_bp.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({'error': 'Method not allowed'}), 405
    
    
    @api_bp.errorhandler(429)
    def rate_limit_exceeded(error):
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': 'Too many requests. Please try again later.'
        }), 429