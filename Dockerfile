# Multi-stage build for Predictive Maintenance MCP Server
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy dependency files first for layer caching
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package (with uvicorn for SSE/HTTP transport)
RUN pip install --no-cache-dir ".[full]" uvicorn

# --- Production stage ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY src/ ./src/
COPY data/ ./data/
COPY resources/ ./resources/
COPY models/ ./models/
COPY reports/ ./reports/

# Create required directories
RUN mkdir -p /app/reports /app/models /app/resources/cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default: SSE transport on 0.0.0.0:8000 (override via env or CLI)
ENV MCP_TRANSPORT=sse
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000

EXPOSE 8000

# Health check (verify imports work)
RUN python -c "from src.machinery_diagnostics_server import mcp; print('Server imports OK')"

ENTRYPOINT ["predictive-maintenance-mcp"]
