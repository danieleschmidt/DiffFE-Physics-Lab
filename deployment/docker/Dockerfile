# Multi-stage Docker build for Physics-Informed Sentiment Analyzer
# Optimized for production deployment with multiple architecture support

# ===== Build Stage =====
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG BUILDARCH
ARG TARGETARCH

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build the package
RUN pip install --no-cache-dir -e .

# ===== Production Stage =====
FROM python:3.11-slim as production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DIFFHE_ENV=production \
    DIFFHE_LOG_LEVEL=INFO

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
WORKDIR /app
COPY --from=builder /app /app

# Set proper ownership
RUN chown -R appuser:appuser /app

# Create directories for runtime data
RUN mkdir -p /app/data /app/logs /app/cache /app/models \
    && chown -R appuser:appuser /app/data /app/logs /app/cache /app/models

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${DIFFHE_API_PORT:-5000}/health || exit 1

# Expose port
EXPOSE 5000

# Set default command
CMD ["python", "-m", "src.api.app"]

# ===== Development Stage =====
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional dev tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Override command for development
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000", "--debug"]