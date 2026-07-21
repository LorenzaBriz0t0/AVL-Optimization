"""Builds a labeled dataset of (design vector -> AVL-derived outputs) via Latin-hypercube
sampling + parallel real-AVL evaluation, for training the NN surrogate (surrogate/model.py)."""
from __future__ import annotations

import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd

from avlnn.config import Constants
from avlnn.design_space import DesignSpace
from avlnn.evaluate import evaluate_design
from avlnn.objective import ObjectiveResult

OUTPUT_COLUMNS = [
    "feasible", "fitness", "l_over_d", "thrust_margin_N",
    "static_margin", "stall_margin_deg",
    "margin_phugoid", "margin_short_period", "margin_dutch_roll", "margin_roll", "margin_spiral",
]


def result_to_row(result: ObjectiveResult) -> dict[str, float]:
    if result.stability is None:
        return {
            "feasible": 0.0, "fitness": result.fitness, "l_over_d": 0.0,
            "thrust_margin_N": 0.0, "static_margin": 0.0, "stall_margin_deg": 0.0,
            "margin_phugoid": 0.0, "margin_short_period": 0.0, "margin_dutch_roll": 0.0,
            "margin_roll": 0.0, "margin_spiral": 0.0,
        }
    s = result.stability
    return {
        "feasible": 1.0,
        "fitness": result.fitness,
        "l_over_d": result.l_over_d,
        "thrust_margin_N": result.thrust_margin_N,
        "static_margin": s.static_margin,
        "stall_margin_deg": s.stall_margin_deg,
        "margin_phugoid": s.mode_checks["phugoid"].margin,
        "margin_short_period": s.mode_checks["short_period"].margin,
        "margin_dutch_roll": s.mode_checks["dutch_roll"].margin,
        "margin_roll": s.mode_checks["roll"].margin,
        "margin_spiral": s.mode_checks["spiral"].margin,
    }


def _evaluate_one(args: tuple[np.ndarray, DesignSpace, Constants]) -> dict[str, float]:
    x, space, constants = args
    design = space.to_dict(x)
    result = evaluate_design(design, constants)
    row = dict(zip(space.order, x))
    row.update(result_to_row(result))
    return row


def generate_dataset(
    space: DesignSpace,
    constants: Constants,
    n_samples: int,
    n_workers: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    samples = space.sample_lhs(n_samples, rng)
    tasks = [(x, space, constants) for x in samples]

    if n_workers <= 1:
        rows = [_evaluate_one(t) for t in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            rows = list(pool.map(_evaluate_one, tasks))

    return pd.DataFrame(rows)


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
