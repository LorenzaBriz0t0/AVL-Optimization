# AVL Aircraft Design Optimization

An evolutionary optimization pipeline for preliminary aircraft design using [Athena Vortex Lattice (AVL)](https://web.mit.edu/drela/Public/web/avl/) as the aerodynamic analysis engine.

The project generates candidate aircraft configurations, evaluates them through AVL, applies aerodynamic, stability, and flight-quality constraints, and uses an evolutionary algorithm to search for improved designs.

---

## Overview

The optimization loop is:

```text
Design Variables
       │
       ▼
Geometry Generation
       │
       ▼
AVL Input Generation
       │
       ▼
      AVL
       │
       ▼
AVL Output Parsing
       │
       ▼
Aerodynamic Evaluation
       │
       ├── Stability / Trim Checks
       ├── Flight-Quality Checks
       └── Performance Metrics
       │
       ▼
Constraint Evaluation
       │
       ▼
Objective Function
       │
       ▼
Evolutionary Algorithm
       │
       └──────────────► New Design
```

The optimization is deliberately coupled directly to AVL rather than using a surrogate model. This makes the aerodynamic evaluation more expensive, but avoids introducing surrogate-model error into the optimization process.

---

## Project Structure

```text
AVL-Optimization/
│
├── config/
│   └── aircraft_spec.yaml
│
├── docker/
│   └── entrypoint.sh
│
├── scripts/
│   ├── make_figures.py
│   ├── run_ea.py
│   └── setup_avl.sh
│
├── src/
│   └── avlnn/
│       ├── avl/
│       ├── geometry/
│       ├── optimization/
│       └── ...
│
├── tests/
│
├── final_avl_files/
├── submission_outputs/
│
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Installation

### Requirements

For local execution, the project requires:

* Python 3.10+
* AVL
* `gfortran`
* X11 libraries
* Xvfb
* Python dependencies listed in `pyproject.toml`

AVL requires an X11 display for some analyses. On headless machines, the project uses `xvfb-run` to provide a virtual display.

---

## Docker

Docker is the recommended way to run the optimization because it provides a reproducible environment containing Python, AVL, Xvfb, and the required system dependencies.

### Build the image

From the repository root:

```bash
docker build -t avl-optimization .
```

To rebuild from scratch without using cached layers:

```bash
docker build --no-cache -t avl-optimization .
```

Verify that the image was created:

```bash
docker images avl-optimization
```

### Run the optimization

The image has a default optimization configuration, so no additional arguments are required:

```bash
docker run --rm avl-optimization
```

The default configuration is:

```text
Population:   40
Generations:  50
Workers:      4
```

The container automatically starts the evolutionary optimization through the project entrypoint.

### Customize the optimization

Arguments passed to `docker run` are forwarded directly to `scripts/run_ea.py`.

For example:

```bash
docker run --rm avl-optimization \
    --population 100 \
    --generations 200 \
    --workers 8
```

Check all available optimization arguments with:

```bash
docker run --rm avl-optimization --help
```

### Interactive shell

To open a shell inside the container instead of running the optimization:

```bash
docker run --rm -it \
    --entrypoint /bin/bash \
    avl-optimization
```

This is useful for debugging the AVL installation or inspecting generated files.

For example:

```bash
which avl
avl
python --version
```

### Mounting output directories

To keep generated results on the host machine, mount the relevant project directories when running the container.

For example:

```bash
docker run --rm \
    -v "$(pwd)/submission_outputs:/app/submission_outputs" \
    -v "$(pwd)/final_avl_files:/app/final_avl_files" \
    avl-optimization
```

Files generated inside these directories will persist on the host after the container exits.

### Running with more workers

The AVL evaluations can be parallelized using multiple workers:

```bash
docker run --rm \
    --cpus 8 \
    avl-optimization \
    --population 100 \
    --generations 200 \
    --workers 8
```

The number of workers should generally be chosen according to the number of CPU cores available to Docker.

### Useful Docker commands

List the image:

```bash
docker images avl-optimization
```

Remove the image:

```bash
docker rmi avl-optimization
```

List running containers:

```bash
docker ps
```

List all containers:

```bash
docker ps -a
```

Clean up stopped containers:

```bash
docker container prune
```

### Typical workflow

A complete workflow from a clean checkout is:

```bash
git clone https://github.com/LorenzaBriz0t0/AVL-Optimization.git
cd AVL-Optimization

docker build -t avl-optimization .

docker run --rm \
    -v "$(pwd)/submission_outputs:/app/submission_outputs" \
    -v "$(pwd)/final_avl_files:/app/final_avl_files" \
    avl-optimization
```

For a larger optimization run:

```bash
docker run --rm \
    --cpus 8 \
    -v "$(pwd)/submission_outputs:/app/submission_outputs" \
    -v "$(pwd)/final_avl_files:/app/final_avl_files" \
    avl-optimization \
    --population 100 \
    --generations 200 \
    --workers 8
```

---

## Local Execution

After installing the package and AVL:

```bash
python scripts/run_ea.py \
    --population 40 \
    --generations 50 \
    --workers 4
```

For headless environments, run the optimization through Xvfb:

```bash
xvfb-run -a python scripts/run_ea.py \
    --population 40 \
    --generations 50 \
    --workers 4
```

---

## Design Variables

The optimizer operates on a vector of aircraft design variables.

These variables define the aircraft geometry and operating configuration used to generate the corresponding AVL model.

The design space and aircraft-level parameters are defined in:

```text
config/aircraft_spec.yaml
```

This keeps the optimization code independent from the specific aircraft configuration.

---

## Aerodynamic Analysis

For each candidate design, the pipeline:

1. Generates the aircraft geometry.
2. Creates the corresponding AVL input files.
3. Runs AVL.
4. Extracts aerodynamic and stability information from AVL.
5. Evaluates the candidate against the optimization constraints.
6. Computes the objective value.

The evolutionary algorithm itself is kept independent of AVL. The optimizer receives an evaluation function and does not need to know how the aerodynamic analysis is performed.

This separation makes it possible to test the optimization algorithm independently from the AVL interface.

---

## Constraints

A candidate aircraft is not evaluated solely according to its objective value.

The optimization framework supports constraints based on aerodynamic and aircraft-level requirements.

### Stability

Examples include:

* Static longitudinal stability
* Directional stability
* Lateral stability
* Trim feasibility
* Stability derivatives

### Flight Quality

Flight-quality checks are being incorporated into the evaluation pipeline.

These checks are intended to determine whether a mathematically feasible aircraft configuration also provides acceptable handling characteristics.

Depending on the flight condition, these may include quantities such as:

* Control authority
* Trim requirements
* Control-surface deflections
* Static margin
* Stability derivatives
* Response characteristics
* Other aircraft-level handling requirements

The exact criteria are defined by the applicable aircraft and flight-quality requirements.

### Performance

Candidate designs can also be evaluated using aerodynamic performance metrics such as:

* Lift
* Drag
* Lift-to-drag ratio
* Required angle of attack
* Trim condition
* Other AVL-derived performance quantities

---

## Optimization

The optimization uses an evolutionary algorithm to explore the aircraft design space.

The general process is:

```text
Initial Population
        │
        ▼
Evaluate Designs
        │
        ▼
Constraint Handling
        │
        ▼
Selection
        │
        ▼
Variation
        │
        ├── Crossover
        └── Mutation
        │
        ▼
New Population
        │
        └──────────────► Evaluation
```

Because each evaluation requires an AVL execution, aerodynamic evaluations are parallelized across worker processes.

For example:

```bash
python scripts/run_ea.py \
    --population 100 \
    --generations 200 \
    --workers 8
```

---

## Objective Function

The objective function combines the desired aerodynamic performance with the aircraft constraints.

A design that produces an attractive aerodynamic objective but violates a required stability or flight-quality constraint is not considered an acceptable solution.

This distinction is important because the optimization problem is not simply:

```text
minimize drag
```

but rather:

```text
optimize aerodynamic performance
subject to aircraft-level constraints
```

---

## Reproducibility

The aircraft specification and optimization configuration are stored in the repository so that experiments can be reproduced.

Results from final optimization runs can be stored under:

```text
submission_outputs/
```

AVL input/output files associated with selected designs can be stored under:

```text
final_avl_files/
```

---

## Development

Install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite with:

```bash
pytest
```

---

## Design Philosophy

The project is organized around a separation between:

* **Aircraft definition**
* **Geometry generation**
* **AVL execution**
* **AVL output parsing**
* **Aerodynamic evaluation**
* **Flight-quality evaluation**
* **Optimization**

This allows the aerodynamic model and aircraft constraints to evolve without requiring changes to the evolutionary algorithm itself.

In particular, the optimizer should remain unaware of whether a constraint comes from AVL, an analytical calculation, or a flight-quality requirement. The evaluation layer is responsible for converting a candidate design into the metrics and constraint violations required by the optimizer.

---

## Current Development

The project is currently focused on extending the aerodynamic optimization framework with **flight-quality constraints**.

The intended evaluation pipeline is therefore:

```text
Aircraft Design
      │
      ▼
Aerodynamic Analysis
      │
      ▼
Stability
      │
      ▼
Trim
      │
      ▼
Flight Quality
      │
      ▼
Performance
      │
      ▼
Feasibility
      │
      ▼
Optimization
```

The goal is to obtain designs that are not merely aerodynamically attractive, but also **stable, controllable, trimmable, and suitable from a flight-quality perspective**.
