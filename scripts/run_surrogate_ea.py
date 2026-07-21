#!/usr/bin/env python3
"""CLI: run the surrogate-assisted EA -- the population is evaluated by the NN surrogate
(microseconds per design), and every few generations the current elite is re-validated
against real AVL, folded into the training set, and the surrogate retrained.

Requires a dataset from run_dataset_gen.py. A trained surrogate (train_surrogate.py) is
loaded if present, otherwise trained from the dataset at startup.

Usage:
    python scripts/run_surrogate_ea.py --population 200 --generations 300

ntfy notifications fire ONLY when a real-AVL re-validation confirms a new best design --
never on surrogate predictions alone. Same topic defaults as run_ea.py.

Outputs (mirroring run_ea.py's archiving):
    runs/surrogate_ea_result.json        latest run (stable path)
    runs/surrogate_ea_<timestamp>.json   one archive per run
    runs/surrogate_log.csv               one summary line per run
    data/dataset_augmented.csv           input dataset + all re-validation rows
    data/surrogate.pt                    updated (actively retrained) surrogate
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from avlnn.config import Constants  # noqa: E402
from avlnn.design_space import DesignSpace  # noqa: E402
from avlnn.ea import EaConfig, GenerationRecord  # noqa: E402
from avlnn.notify import resolve_ntfy_config, send_ntfy  # noqa: E402
from avlnn.objective import ObjectiveResult  # noqa: E402
from avlnn.surrogate.dataset import load_dataset, save_dataset  # noqa: E402
from avlnn.surrogate.model import SurrogateBundle, train_surrogate  # noqa: E402
from avlnn.surrogate.surrogate_ea import SurrogateEaConfig, run_surrogate_assisted_ea  # noqa: E402

HIGHLIGHT_KEYS = (
    "wing_aspect_ratio", "wing_taper_ratio", "wing_sweep_c4_deg",
    "wing_dihedral_deg", "wing_area_m2",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--population", type=int, default=200)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--revalidate-every", type=int, default=5,
        help="run this many surrogate generations between real-AVL re-validation rounds",
    )
    parser.add_argument(
        "--revalidate-top", type=int, default=5,
        help="how many of the block's best designs get re-validated with real AVL",
    )
    parser.add_argument("--retrain-epochs", type=int, default=100)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data" / "dataset.csv")
    parser.add_argument("--surrogate", type=Path, default=REPO_ROOT / "data" / "surrogate.pt")
    parser.add_argument(
        "--dataset-out", type=Path, default=REPO_ROOT / "data" / "dataset_augmented.csv",
        help="where to save the dataset grown by re-validation rows",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "surrogate_ea_result.json")
    parser.add_argument("--ntfy-topic", type=str, default=None,
                        help="see run_ea.py; '' disables notifications")
    parser.add_argument("--ntfy-server", type=str, default=None)
    args = parser.parse_args()

    constants = Constants.load()
    space = DesignSpace.load()

    if not args.dataset.exists():
        sys.exit(
            f"dataset not found: {args.dataset}\n"
            "generate one first:  python scripts/run_dataset_gen.py --samples 2000 --workers 8"
        )
    dataset = load_dataset(args.dataset)
    print(f"Loaded dataset: {len(dataset)} rows from {args.dataset}")

    if args.surrogate.exists():
        bundle = SurrogateBundle.load(args.surrogate)
        print(f"Loaded surrogate from {args.surrogate}")
    else:
        print(f"No surrogate at {args.surrogate}; training one from the dataset first...")
        bundle, metrics = train_surrogate(dataset, space)
        print(f"Trained (validation MSE {metrics['val_mse']:.5f})")

    ntfy_config = resolve_ntfy_config(args.ntfy_topic, args.ntfy_server)
    if ntfy_config:
        print(f"ntfy notifications (real-AVL-confirmed bests only) -> "
              f"{ntfy_config.server}/{ntfy_config.topic}")
    else:
        print("ntfy notifications disabled")

    # The surrogate evaluates in-process in microseconds; n_workers stays 1 because the
    # closure over the model isn't picklable for ProcessPoolExecutor and wouldn't pay anyway.
    config = SurrogateEaConfig(
        ea_config=EaConfig(
            population_size=args.population, n_generations=args.generations,
            n_workers=1, seed=args.seed,
        ),
        revalidate_every_n_generations=args.revalidate_every,
        n_elite_to_revalidate=args.revalidate_top,
        retrain_epochs=args.retrain_epochs,
    )

    best_real_fitness = -math.inf
    best_real_design: dict[str, float] | None = None
    revalidation_history: list[dict] = []

    def on_generation(record: GenerationRecord) -> None:
        if record.generation % 10 == 0 or record.generation == args.generations - 1:
            print(
                f"gen {record.generation:4d}  surrogate best={record.best_fitness:9.4f}  "
                f"mean={record.mean_fitness:9.4f}"
            )

    def on_revalidate(x: np.ndarray, real: ObjectiveResult) -> None:
        nonlocal best_real_fitness, best_real_design
        design = space.to_dict(x)
        revalidation_history.append({
            "design": design,
            "fitness": real.fitness,
            "l_over_d": real.l_over_d,
            "all_passed": real.stability.all_passed if real.stability else False,
            "infeasible_reason": real.infeasible_reason,
        })
        if real.fitness > best_real_fitness:
            best_real_fitness = real.fitness
            best_real_design = design
            print(f"    real-AVL confirmed new best: fitness={real.fitness:.4f}")
            if ntfy_config:
                highlights = ", ".join(f"{k}={design[k]:.3g}" for k in HIGHLIGHT_KEYS)
                send_ntfy(
                    ntfy_config,
                    message=f"AVL-confirmed fitness = {real.fitness:.4f}\n{highlights}",
                    title="Surrogate EA: new best design (real AVL)",
                    tags=["airplane", "white_check_mark"],
                )

    started_at = datetime.datetime.now()
    result, bundle, dataset = run_surrogate_assisted_ea(
        space, constants, bundle, dataset, config,
        on_generation=on_generation, on_revalidate=on_revalidate,
    )

    save_dataset(dataset, args.dataset_out)
    bundle.save(args.surrogate)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "settings": {
            "population": args.population, "generations": args.generations,
            "seed": args.seed, "revalidate_every": args.revalidate_every,
            "revalidate_top": args.revalidate_top,
        },
        "best_real_fitness": None if best_real_design is None else best_real_fitness,
        "best_real_design": best_real_design,
        "best_surrogate_fitness": result.best_fitness,
        "best_surrogate_design": space.to_dict(result.best_design),
        "revalidations": revalidation_history,
    }
    text = json.dumps(payload, indent=2)
    args.out.write_text(text, encoding="utf-8")
    archive = args.out.parent / f"surrogate_ea_{started_at:%Y%m%d_%H%M%S}.json"
    archive.write_text(text, encoding="utf-8")

    log_path = args.out.parent / "surrogate_log.csv"
    if not log_path.exists():
        log_path.write_text(
            "started_at,population,generations,seed,best_real_fitness,archive\n", encoding="utf-8",
        )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{started_at.isoformat(timespec='seconds')},{args.population},{args.generations},"
            f"{args.seed},{best_real_fitness:.6f},{archive.name}\n"
        )

    print(f"\nAugmented dataset ({len(dataset)} rows) -> {args.dataset_out}")
    print(f"Updated surrogate -> {args.surrogate}")
    if best_real_design is None:
        print("WARNING: no re-validated design was feasible; inspect the revalidations "
              f"in {args.out}")
    else:
        print(f"Best real-AVL-confirmed fitness: {best_real_fitness:.4f}")
    print(f"Result written to {args.out} (archived as {archive.name})")


if __name__ == "__main__":
    main()
