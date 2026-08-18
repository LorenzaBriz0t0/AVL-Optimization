FROM python:3.10-slim

# System dependencies required to build and run AVL.
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libx11-dev \
    libxext-dev \
    xvfb \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the project.
COPY . .

# Build AVL.
# setup_avl.sh assumes dependencies are already installed.
RUN bash scripts/setup_avl.sh

# Put the AVL binary on PATH.
RUN BIN_PATH=$(find /app/vendor -name "avl" -type f -executable | head -n 1) && \
    if [ -z "$BIN_PATH" ]; then \
        echo "AVL binary not found!" && \
        exit 1; \
    fi && \
    ln -s "$BIN_PATH" /usr/local/bin/avl

ENV AVL_BIN="/usr/local/bin/avl"
ENV PYTHONUNBUFFERED=1

# Install the Python package and development dependencies.
RUN pip install --no-cache-dir ".[dev]"

# Run through the wrapper so xvfb-run is not PID 1.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["--population", "40", "--generations", "50", "--workers", "4"]