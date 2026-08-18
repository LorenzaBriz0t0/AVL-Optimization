#!/usr/bin/env python3
"""CLI: run the evolutionary algorithm with real AVL evaluations in the loop.

Usage:
    python scripts/run_ea.py --population 40 --generations 50 --workers 4

ntfy notifications on every new best design are ON by default (topic 'avlnn-pipeline' on
ntfy.sh -- ntfy topics are public, so set your own unguessable topic via --ntfy-topic/
--ntfy-server or the NTFY_TOPIC/NTFY_SERVER env vars); disable with --ntfy-topic ''.
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
from avlnn.ea import EaConfig, GenerationRecord, run_ea  # noqa: E402
from avlnn.evaluate import evaluate_design  # noqa: E402
from avlnn.notify import (  # noqa: E402
    DEFAULT_NTFY_SERVER,
    DEFAULT_NTFY_TOPIC,
    NTFY_SERVER_ENV_VAR,
    NTFY_TOPIC_ENV_VAR,
    resolve_ntfy_config,
    send_ntfy,
)


class AvlFitnessFn:
    """Picklable evaluate_fn (required for ProcessPoolExecutor): design array -> fitness via AVL."""

    def __init__(self, space: DesignSpace, constants: Constants):
        self.space = space
        self.constants = constants

    def __call__(self, x: np.ndarray) -> float:
        design = self.space.to_dict(x)
        return evaluate_design(design, self.constants).fitness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "ea_result.json")
    parser.add_argument(
        "--ntfy-topic", type=str, default=None,
        help="ntfy topic to push a notification to on every new best design. "
             f"Precedence: this flag, then the {NTFY_TOPIC_ENV_VAR} env var, then "
             f"'{DEFAULT_NTFY_TOPIC}'. Pass an empty string ('') to disable notifications.",
    )
    parser.add_argument(
        "--ntfy-server", type=str, default=None,
        help=f"ntfy server (default {DEFAULT_NTFY_SERVER}, or the {NTFY_SERVER_ENV_VAR} env var)",
    )
    args = parser.parse_args()
    print("=" * 60, flush=True)
    print("AVL Aircraft Design Optimization", flush=True)
    print("=" * 60, flush=True)
    print(f"Population:    {args.population}", flush=True)
    print(f"Generations:   {args.generations}", flush=True)
    print(f"Workers:       {args.workers}", flush=True)
    print(flush=True)
    print("Starting evolutionary optimization...", flush=True)

    constants = Constants.load()
    space = DesignSpace.load()
    fitness_fn = AvlFitnessFn(space, constants)
    config = EaConfig(
        population_size=args.population, n_generations=args.generations,
        n_workers=args.workers, seed=args.seed,
    )

    ntfy_config = resolve_ntfy_config(args.ntfy_topic, args.ntfy_server)
    if ntfy_config:
        print(f"ntfy notifications enabled -> {ntfy_config.server}/{ntfy_config.topic}")
    else:
        print("ntfy notifications disabled")

    best_fitness_so_far = -math.inf

    def on_generation(record: GenerationRecord) -> None:
        nonlocal best_fitness_so_far
        print(
            f"gen {record.generation:3d}  "
            f"best={record.best_fitness:8.4f}  mean={record.mean_fitness:8.4f}"
        )
        if ntfy_config and record.best_fitness > best_fitness_so_far:
            best_fitness_so_far = record.best_fitness
            design = space.to_dict(record.best_design)
            highlights = ", ".join(
                f"{k}={design[k]:.3g}" for k in (
                    "wing_aspect_ratio", "wing_taper_ratio", "wing_sweep_c4_deg",
                    "wing_dihedral_deg", "wing_area_m2",
                )
            )
            send_ntfy(
                ntfy_config,
                message=f"Generation {record.generation}: fitness = {record.best_fitness:.4f}\n{highlights}",
                title="AVL EA: new best design found",
                tags=["airplane"],
            )

    started_at = datetime.datetime.now()
    result = run_ea(space, fitness_fn, config, on_generation=on_generation)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "settings": {
            "population": args.population, "generations": args.generations,
            "workers": args.workers, "seed": args.seed,
        },
        "best_fitness": result.best_fitness,
        "best_design": space.to_dict(result.best_design),
        "history": [
            {"generation": r.generation, "best_fitness": r.best_fitness, "mean_fitness": r.mean_fitness}
            for r in result.history
        ],
    }
    text = json.dumps(payload, indent=2)

    # args.out always holds the LATEST result (stable path for downstream tools); every run
    # is also archived under a timestamped name and summarized in runs/log.csv, so no run is
    # ever silently overwritten.
    args.out.write_text(text, encoding="utf-8")
    archive = args.out.parent / f"ea_{started_at:%Y%m%d_%H%M%S}.json"
    archive.write_text(text, encoding="utf-8")

    log_path = args.out.parent / "log.csv"
    if not log_path.exists():
        log_path.write_text("started_at,population,generations,seed,best_fitness,archive\n", encoding="utf-8")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{started_at.isoformat(timespec='seconds')},{args.population},{args.generations},"
            f"{args.seed},{result.best_fitness:.6f},{archive.name}\n"
        )

    print(f"\nBest fitness: {result.best_fitness:.4f}")
    print(f"Best design written to {args.out} (archived as {archive.name}, logged in {log_path.name})")


if __name__ == "__main__":
    main()
