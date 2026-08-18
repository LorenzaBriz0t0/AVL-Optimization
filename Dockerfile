FROM python:3.10-slim

# 1. Install system dependencies for the build script
RUN apt-get update && apt-get install -y \
    sudo \
    build-essential \
    gfortran \
    libx11-dev \
    libxext-dev \
    xvfb \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy the entire repository
COPY . /app

# 3. Run the provided setup script
RUN bash scripts/setup_avl.sh

# 4. Safely link the binary
RUN BIN_PATH=$(find /app -name "avl" -type f -executable | head -n 1) && \
    if [ -z "$BIN_PATH" ]; then echo "❌ AVL binary not found!"; exit 1; fi && \
    ln -s "$BIN_PATH" /usr/local/bin/avl

ENV AVL_BIN="/usr/local/bin/avl"

# 5. Install Python requirements (Now nice and lightweight!)
RUN pip install --no-cache-dir ".[dev]"

# 6. Set up ENTRYPOINT
ENTRYPOINT ["xvfb-run", "-a", "python"]
CMD ["scripts/run_ea.py", "--population", "40", "--generations", "50", "--workers", "4"]