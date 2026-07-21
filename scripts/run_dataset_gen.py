#!/usr/bin/env python3
"""CLI: generate a surrogate training dataset via LHS sampling + parallel real-AVL evaluation.

Usage:
    python scripts/run_dataset_gen.py --samples 2000 --workers 8 --out data/dataset.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from avlnn.config import Constants  # noqa: E402
from avlnn.design_space import DesignSpace  # noqa: E402
from avlnn.surrogate.dataset import generate_dataset, save_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "dataset.csv")
    args = parser.parse_args()

    constants = Constants.load()
    space = DesignSpace.load()

    df = generate_dataset(space, constants, args.samples, n_workers=args.workers, seed=args.seed)
    save_dataset(df, args.out)

    n_feasible = int(df["feasible"].sum())
    print(f"Generated {len(df)} samples ({n_feasible} feasible) -> {args.out}")


if __name__ == "__main__":
    main()
