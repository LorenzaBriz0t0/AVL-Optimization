# AVL Neural-Network + Evolutionary-Algorithm Aircraft Design Pipeline

An automated design-search pipeline for a fixed-wing aircraft that must be aerodynamically
efficient and both statically and dynamically stable. [AVL](https://web.mit.edu/drela/Public/web/avl/)
(Athena Vortex Lattice) is the aerodynamic/stability ground truth, a small PyTorch surrogate model
learns to approximate AVL's outputs for speed, and a real-valued evolutionary algorithm (EA)
searches the design space against both — every design that survives is re-validated against real
AVL before it's trusted.

Originally built for a university course project (design a jet trainer: m = 4000 kg, cruise
Mach 0.5 at 11 km, NACA 2312 wing, flat-plate tails, 10° stall limit), the pipeline itself is not
specific to that aircraft — the fixed spec lives entirely in `config/aircraft_constants.yaml` and
the free design-variable bounds in `config/design_space.yaml`, so a different airframe is a config
change, not a code change.

## How it works

Every design vector — an 18-parameter `dict[str, float]` (wing AR/taper/sweep/dihedral/
incidence/area/location, aileron geometry, horizontal- and vertical-tail sizing, fuselage mass
layout) — passes through one function, `evaluate_design()`, which:

1. **Resolves geometry** (`derived.py`) — pure math, no AVL: wing/tail planform, span, chords,
   and the aircraft CG (a closed-form function of the wing mass points' location).
2. **Renders AVL input files** (`geometry.py`, `massfile.py`, `runcase.py`) — the `.avl`/`.mass`/
   `.run` text formats AVL expects.
3. **Runs AVL** (`avl_driver.py`) — drives the AVL binary as a subprocess via a scripted stdin
   command sequence (AVL has no non-interactive batch mode) in a scratch directory.
4. **Parses the output** (`parse.py`) — regexes AVL's stdout and `.eig` eigenvalue dump into
   structured stability derivatives and eigenvalues.
5. **Classifies stability** (`stability.py`) — sorts the eigenvalues into the five classical
   dynamic modes (phugoid, short period, dutch roll, roll, spiral) and checks every requirement:
   static margin, stall margin (checked as trim angle *anywhere on the wing*, not just body
   alpha), and every mode's stability — spiral is allowed to be unstable if its time constant is
   long enough.
6. **Scores it** (`objective.py`) — cruise L/D minus proportional penalties for any failed
   constraint, so the EA has a gradient to climb even from infeasible designs, plus a small
   saturating bonus that breaks ties among equally efficient designs in favor of larger stability
   margins.

`ea.py` is deliberately AVL-agnostic — it only needs an `evaluate_fn(x) -> float` callable — which
is what lets the identical EA engine run in two modes:

- **Pure-AVL search** (`scripts/run_ea.py`): every fitness evaluation is a real AVL run. Slower,
  but the ground truth.
- **Surrogate-assisted search** (`scripts/run_surrogate_ea.py`): the EA evaluates against a
  PyTorch MLP trained on AVL-generated data instead, with periodic real-AVL re-validation of the
  elite population and active-learning retraining. Meant to be faster; see [Results](#results) for
  why that didn't pan out here.

## Repository layout

```
config/            fixed spec constants + free design-variable bounds (YAML)
src/avlnn/          the pipeline: geometry/mass/run-case generation, AVL driver, parsing,
                     stability classification, objective, the EA, the NN surrogate
scripts/            CLIs: dataset generation, surrogate training, both EA modes, an
                     unattended end-to-end pipeline script
tests/               pytest suite covering file generation, parsing, and stability/EA logic
                     in isolation (no AVL binary required)
final_avl_files/    one converged design's AVL input/output files, reproducible via the
                     included Instruction.txt
submission_outputs/ an archived pure-AVL EA run, dataset, trained surrogate, and
                     surrogate-EA run, kept as a worked example
```

## Setup

```bash
# Build AVL from source (no prebuilt package exists); prints an AVL_BIN path to export
bash scripts/setup_avl.sh
export AVL_BIN=...

pip install -e ".[dev]"
pytest   # full suite is pure Python; doesn't need the AVL binary
```

## Usage

```bash
# Pure-AVL evolutionary search (real AVL call every fitness evaluation)
python scripts/run_ea.py --population 40 --generations 50 --workers 4

# Build a training dataset via real AVL, then train the surrogate
python scripts/run_dataset_gen.py --samples 2000 --workers 8 --out data/dataset.csv
python scripts/train_surrogate.py --dataset data/dataset.csv --out data/surrogate.pt

# Surrogate-assisted EA (trains the surrogate itself if missing)
python scripts/run_surrogate_ea.py --population 200 --generations 300

# Unattended dataset -> train -> surrogate EA -> pure-AVL cross-check, survives disconnects
nohup bash scripts/run_overnight.sh >/dev/null 2>&1 &
```

All scripts can optionally push a notification via [ntfy](https://ntfy.sh) on every
AVL-confirmed improvement (`--ntfy-topic your-own-topic`; ntfy topics are public, so pick your
own unguessable one rather than relying on the default).

## Results

For the jet-trainer spec the pipeline was built against, the EA converged — from four independent
random seeds, to the same design — to the closed-form optimum implied by the assignment's drag
model (max L/D depends only on aspect ratio and wing area): **L/D = 14.07**, static margin 0.36,
stall margin 3.2°, and every dynamic mode stable except spiral (allowed, since its time constant
of ~730 s is far past the 20 s minimum).

The surrogate is included as an honest negative result, not a success story: with ~2000 training
samples the network's mean error over near-feasible designs was too large to resolve the narrow
feasible ridge in the design space, and the surrogate-assisted EA converged to a design well below
the pure-AVL optimum. The real-AVL re-validation step did its job — every reported improvement was
independently confirmed — but the surrogate itself didn't accelerate the search here. Worth reading
`src/avlnn/surrogate/` as a demonstration of the active-learning *plumbing* rather than as a tuned
model.

## License

MIT — see `LICENSE`.
