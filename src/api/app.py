"""Flask application factory for DiffFE-Physics-Lab API."""

import logging
import os
from typing import Optional

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS

    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

from .error_handlers import register_error_handlers
from .middleware import setup_middleware
from .routes import register_routes


def create_app(config: Optional[dict] = None) -> "Flask":
    """Create and configure Flask application.

    Parameters
    ----------
    config : dict, optional
        Application configuration, by default None

    Returns
    -------
    Flask
        Configured Flask application

    Raises
    ------
    ImportError
        If Flask is not available
    """
    if not HAS_FLASK:
        raise ImportError("Flask API requires Flask: pip install flask flask-cors")

    app = Flask(__name__)

    # Default configuration
    app.config.update(
        {
            "SECRET_KEY": os.getenv("DIFFHE_SECRET_KEY", "dev-secret-key"),
            "DEBUG": os.getenv("DIFFHE_DEBUG", "false").lower() == "true",
            "TESTING": False,
            "MAX_CONTENT_LENGTH": 100 * 1024 * 1024,  # 100MB max file size
            "UPLOAD_FOLDER": os.getenv("DIFFHE_UPLOAD_PATH", "./uploads"),
            "RESULT_FOLDER": os.getenv("DIFFHE_RESULT_PATH", "./results"),
            "CACHE_TYPE": "simple",
            "CACHE_DEFAULT_TIMEOUT": 300,
            "RATELIMIT_STORAGE_URL": "memory://",
        }
    )

    # Override with provided config
    if config:
        app.config.update(config)

    # Setup CORS
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": ["*"],
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
            }
        },
    )

    # Setup logging
    if not app.debug:
        logging.basicConfig(level=logging.INFO)

    # Setup middleware
    setup_middleware(app)

    # Register routes
    register_routes(app)

    # Register error handlers
    register_error_handlers(app)

    # Health check endpoint
    @app.route("/health")
    def health_check():
        """Basic health check endpoint."""
        return jsonify(
            {
                "status": "healthy",
                "version": "1.0.0-dev",
                "timestamp": request.environ.get("REQUEST_TIME", ""),
                "backends": _check_backends(),
            }
        )

    # API info endpoint
    @app.route("/api/info")
    def api_info():
        """API information endpoint."""
        return jsonify(
            {
                "name": "DiffFE-Physics-Lab API",
                "version": "1.0.0-dev",
                "description": "REST API for differentiable finite element computations",
                "documentation": "/api/docs",
                "endpoints": {
                    "problems": "/api/problems",
                    "solutions": "/api/solutions",
                    "meshes": "/api/meshes",
                    "operators": "/api/operators",
                    "optimization": "/api/optimization",
                },
                "features": {
                    "firedrake": _check_firedrake(),
                    "jax": _check_jax(),
                    "torch": _check_torch(),
                    "gpu": _check_gpu(),
                },
            }
        )

    return app


def _check_backends() -> dict:
    """Check availability of AD backends."""
    backends = {}

    try:
        import jax

        backends["jax"] = {
            "available": True,
            "version": jax.__version__,
            "devices": [str(d) for d in jax.devices()],
        }
    except ImportError:
        backends["jax"] = {"available": False}

    try:
        import torch

        backends["torch"] = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": (
                torch.cuda.is_available()
                if hasattr(torch.cuda, "is_available")
                else False
            ),
        }
    except ImportError:
        backends["torch"] = {"available": False}

    return backends


def _check_firedrake() -> bool:
    """Check if Firedrake is available."""
    try:
        import firedrake

        return True
    except ImportError:
        return False


def _check_jax() -> bool:
    """Check if JAX is available."""
    try:
        import jax

        return True
    except ImportError:
        return False


def _check_torch() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch

        return True
    except ImportError:
        return False


def _check_gpu() -> bool:
    """Check if GPU acceleration is available."""
    # Check JAX GPU
    try:
        import jax

        if len(jax.devices("gpu")) > 0:
            return True
    except:
        pass

    # Check PyTorch GPU
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except:
        pass

    return False


# Create default app instance
app = None
if HAS_FLASK:
    app = create_app()


if __name__ == "__main__":
    if not HAS_FLASK:
        print("Flask not available. Install with: pip install flask flask-cors")
        exit(1)

    # Development server
    port = int(os.getenv("DIFFHE_API_PORT", 5000))
    debug = os.getenv("DIFFHE_DEBUG", "false").lower() == "true"

    print(f"Starting DiffFE-Physics-Lab API server on port {port}")
    print(f"Debug mode: {debug}")
    print(f"API documentation: http://localhost:{port}/api/docs")

    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
