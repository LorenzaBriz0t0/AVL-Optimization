#!/usr/bin/env python3
"""Generate the report figures from archived run data (run on the server, where runs/ and
data/ live). Writes SVG (vector, for the report) + PNG (preview) into figures/.

    pip install matplotlib   # one-time; deliberately not a project dependency
    python scripts/make_figures.py

Figures:
  fig_ea_convergence     best fitness vs generation, every archived pure-AVL EA run
  fig_surrogate_parity   NN-predicted vs real-AVL fitness over the training dataset
  fig_surrogate_reval    real-AVL fitness of surrogate-elite designs per re-validation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Okabe-Ito hues (colorblind-safe); identity is never color-alone -- the highlighted
# series is also the only solid/wide line, and plateaus carry direct labels.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#8A8A8A"
INK = "#333333"

FIGURES_DIR = REPO_ROOT / "figures"


def _style(ax):
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    for item in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        item.set_color(INK)


def _save(fig, name: str):
    FIGURES_DIR.mkdir(exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.svg + .png")


def fig_ea_convergence():
    runs = sorted(REPO_ROOT.glob("runs/ea_2*.json"))
    if not runs:
        print("no runs/ea_*.json archives found -- skipping convergence figure")
        return
    final = json.loads((REPO_ROOT / "final_design.json").read_text())
    final_hist = [(h["generation"], h["best_fitness"]) for h in final["history"]]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for path in runs:
        data = json.loads(path.read_text())
        hist = [(h["generation"], h["best_fitness"]) for h in data["history"]]
        if hist == final_hist:
            continue  # drawn separately below
        g, b = zip(*hist)
        ax.plot(g, b, color=GRAY, linewidth=1.0, linestyle="--", alpha=0.7)

    g, b = zip(*final_hist)
    ax.plot(g, b, color=BLUE, linewidth=2.0, label="final run")
    # No numeric label: the run-time fitness includes the margin bonus as computed during
    # the run (pre density-fix), which differs in the third decimal from the re-verified
    # value -- quoting a number here would fight the report's tables.
    ax.annotate("final design", xy=(g[-1], b[-1]),
                xytext=(-8, -14), textcoords="offset points", ha="right",
                fontsize=9, color=BLUE)

    ax.plot([], [], color=GRAY, linewidth=1.0, linestyle="--", label="earlier runs")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (L/D + margin bonus)")
    ax.set_title("Evolutionary-algorithm convergence (real-AVL fitness)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    _save(fig, "fig_ea_convergence")


def fig_surrogate_parity():
    import pandas as pd
    from avlnn.design_space import DesignSpace
    from avlnn.surrogate.model import SurrogateBundle

    dataset_path = REPO_ROOT / "data" / "dataset_augmented.csv"
    model_path = REPO_ROOT / "data" / "surrogate.pt"
    if not dataset_path.exists() or not model_path.exists():
        print("dataset_augmented.csv or surrogate.pt missing -- skipping parity figure")
        return

    space = DesignSpace.load()
    bundle = SurrogateBundle.load(model_path)
    df = pd.read_csv(dataset_path)
    x = df[list(space.order)].to_numpy()
    actual = df["fitness"].to_numpy()
    predicted = bundle.predict(x)[:, bundle.output_order.index("fitness")]

    # Feasible-region view: the -100s/-1000s penalty tail compresses everything
    # interesting into a corner, so clip the axes to the near-feasible band and
    # report how many points are outside it.
    lo, hi = -30.0, 20.0
    near = actual > lo                       # near-feasible designs (penalty tail excluded)
    in_view = near & (predicted > lo) & (predicted < hi)
    mae_near = float(np.mean(np.abs(predicted[near] - actual[near])))

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(actual[in_view], predicted[in_view], s=8, alpha=0.35, color=BLUE,
               edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color=INK, linewidth=0.8)
    ax.annotate("perfect prediction", xy=(hi, hi), xytext=(-10, -16),
                textcoords="offset points", ha="right", fontsize=8, color=INK)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Real AVL fitness")
    ax.set_ylabel("NN-predicted fitness")
    ax.set_title("Surrogate prediction vs real AVL")
    ax.text(0.03, 0.97,
            f"{int(near.sum())} near-feasible designs shown\n"
            f"MAE over them = {mae_near:.1f}\n"
            f"({int(near.sum() - in_view.sum())} predictions beyond axes; "
            f"{int((~near).sum())} penalty-tail designs excluded)",
            transform=ax.transAxes, va="top", fontsize=8, color=INK)
    ax.set_aspect("equal")
    _style(ax)
    _save(fig, "fig_surrogate_parity")


def fig_surrogate_reval():
    path = REPO_ROOT / "runs" / "surrogate_ea_result.json"
    if not path.exists():
        print("surrogate_ea_result.json missing -- skipping re-validation figure")
        return
    reval = json.loads(path.read_text())["revalidations"]
    fitness = np.array([r["fitness"] for r in reval])
    best_so_far = np.maximum.accumulate(fitness)
    idx = np.arange(1, len(fitness) + 1)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    shown = fitness > -30
    ax.scatter(idx[shown], fitness[shown], s=8, alpha=0.35, color=ORANGE,
               edgecolors="none", label="re-validated elite")
    ax.plot(idx, best_so_far, color=BLUE, linewidth=2.0, label="best confirmed so far")
    ax.annotate(f"{best_so_far[-1]:.3f}", xy=(idx[-1], best_so_far[-1]),
                xytext=(-6, -14), textcoords="offset points", ha="right",
                fontsize=9, color=BLUE)
    # Axis hugs the data: every re-validated elite was feasible, so there is no penalty
    # tail to reserve space for (that fact belongs in the caption, not in empty axes).
    n_hidden = int((~shown).sum())
    if n_hidden:
        ax.text(0.02, 0.03, f"{n_hidden} infeasible checks below axis",
                transform=ax.transAxes, fontsize=8, color=INK)
    ax.margins(y=0.12)
    ax.set_xlabel("Real-AVL re-validation (in run order)")
    ax.set_ylabel("Real AVL fitness")
    ax.set_title("Surrogate EA: real-AVL checks of elite designs")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    _save(fig, "fig_surrogate_reval")


if __name__ == "__main__":
    fig_ea_convergence()
    fig_surrogate_parity()
    fig_surrogate_reval()
