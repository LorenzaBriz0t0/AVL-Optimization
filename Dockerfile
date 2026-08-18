FROM python:3.10-slim

# Install system dependencies for the build script
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
COPY . /app

# Run the provided setup script (this actually works!)
RUN bash scripts/setup_avl.sh

# Find the compiled 'avl' binary (whatever folder it ended up in) and link it to a guaranteed path
RUN find /app/vendor -name "avl" -type f -executable -exec ln -s {} /usr/local/bin/avl \;

# Point the Python scripts to our guaranteed path
ENV AVL_BIN="/usr/local/bin/avl"

# Install Python requirements
RUN pip install --no-cache-dir -e ".[dev]"

# Set up ENTRYPOINT
ENTRYPOINT ["xvfb-run", "-a", "python"]
CMD ["scripts/run_ea.py", "--population", "40", "--generations", "50", "--workers", "4"]