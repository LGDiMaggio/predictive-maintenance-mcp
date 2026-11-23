# Multi-stage build for Predictive Maintenance MCP Server
#
# TORCH_VARIANT selects how the dependencies are installed:
#   default  ".[full]" and uvicorn, resolved when the image is built (local builds)
#   cpu      docker/requirements-cpu.txt: the uv.lock pins of the full extra
#            without any NVIDIA, CUDA or Triton component, hash-checked; the
#            official image uses it (see tools/cpu_requirements.py)
ARG TORCH_VARIANT=default

# python:3.11-slim (Python 3.11.18, Debian trixie), pinned by its index digest
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 AS builder
ARG TORCH_VARIANT

WORKDIR /app

# Copy dependency files first for layer caching. The LICENSE goes into the
# installed package's metadata; README.md is the package's long description.
COPY pyproject.toml README.md LICENSE ./
COPY docker/requirements-cpu.txt ./docker/requirements-cpu.txt
COPY src/ ./src/

# Install the package (with uvicorn for SSE/HTTP transport). The cpu variant
# installs exactly the pinned files, then the project without dependencies
# (built with the pinned setuptools), and checks that nothing is missing.
RUN set -eu; \
    case "$TORCH_VARIANT" in \
      cpu) \
        pip install --no-cache-dir --require-hashes --no-deps -r docker/requirements-cpu.txt; \
        pip install --no-cache-dir --no-deps --no-build-isolation .; \
        pip check ;; \
      default) \
        pip install --no-cache-dir --upgrade pip setuptools wheel; \
        pip install --no-cache-dir ".[full]" uvicorn ;; \
      *) \
        echo "TORCH_VARIANT must be cpu or default, not '$TORCH_VARIANT'" >&2; \
        exit 1 ;; \
    esac

# --- Production stage ---
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

WORKDIR /app

# Copy installed packages from builder. The builder's site-packages replaces the
# base image's own (pip, setuptools, wheel, packaging) instead of merging into
# it, so no second dist-info of another version survives next to the new one.
RUN rm -rf /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files: the server, the project's license and the bearing
# catalog, each by its exact path. Signals, trained models, reports and
# machine manuals are not copied; docker-compose.yml mounts them at run time.
COPY src/ ./src/
COPY LICENSE ./LICENSE
COPY resources/bearing_catalogs/README.md resources/bearing_catalogs/common_bearings_catalog.json ./resources/bearing_catalogs/

# Create the folders the server and the mounts expect
RUN mkdir -p /app/data /app/models /app/reports /app/resources/cache /app/resources/machine_manuals

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default: SSE transport on 0.0.0.0:8000 (override via env or CLI)
ENV MCP_TRANSPORT=sse
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000

EXPOSE 8000

# Build-time check: the installed package must import cleanly
RUN python -c "from predictive_maintenance_mcp.server import mcp; print('Server imports OK')"

# Runtime health check: the SSE/HTTP port must accept connections
# (shell form so $MCP_PORT is expanded; irrelevant for stdio deployments)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, socket; socket.create_connection(('127.0.0.1', int(os.environ.get('MCP_PORT', '8000'))), timeout=3)" || exit 1

ENTRYPOINT ["predictive-maintenance-mcp"]
