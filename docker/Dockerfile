FROM python:3.11-slim AS base

LABEL maintainer="DiffFE-Physics-Lab Team <team@diffhe-physics.org>"
LABEL description="Differentiable Finite Element Framework for Physics-Informed Machine Learning"
LABEL version="1.0.0-dev"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 diffhe && \
    useradd --uid 1000 --gid diffhe --shell /bin/bash --create-home diffhe

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements*.txt ./
COPY setup.py ./
COPY pyproject.toml* ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Development stage
FROM base AS development

# Install development dependencies
RUN pip install --no-cache-dir -e .[dev,docs,performance]

# Copy source code
COPY --chown=diffhe:diffhe . .

# Switch to non-root user
USER diffhe

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "src.api.app"]

# Production stage
FROM base AS production

# Copy only necessary files
COPY --chown=diffhe:diffhe src/ ./src/
COPY --chown=diffhe:diffhe setup.py ./
COPY --chown=diffhe:diffhe README.md ./
COPY --chown=diffhe:diffhe LICENSE ./

# Install production dependencies only
RUN pip install --no-cache-dir -e .[minimal]

# Switch to non-root user
USER diffhe

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["python", "-m", "src.api.app", "--production"]

# Testing stage
FROM development AS testing

# Run tests
RUN python -m pytest tests/ --tb=short -v

# GPU-enabled stage
FROM nvidia/cuda:11.8-devel-ubuntu20.04 AS gpu

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install
COPY requirements*.txt ./
COPY setup.py ./
RUN pip install --no-cache-dir -e .[gpu,torch,jax]

# Copy source
COPY --chown=1000:1000 . .

# Create non-root user
RUN useradd --uid 1000 --create-home diffhe
USER diffhe

# GPU command
CMD ["python", "-m", "src.api.app", "--gpu"]