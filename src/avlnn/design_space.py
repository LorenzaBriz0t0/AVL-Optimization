"""Design-vector bounds, dict<->array conversion, and sampling for the EA and NN surrogate."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import yaml

from avlnn.config import CONFIG_DIR


@dataclasses.dataclass(frozen=True)
class DesignSpace:
    order: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def load(cls, path: Path | None = None) -> "DesignSpace":
        with open(path or CONFIG_DIR / "design_space.yaml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        order = tuple(data["order"])
        variables = data["variables"]
        missing = set(order) - set(variables)
        if missing:
            raise ValueError(f"design_space.yaml 'order' lists unknown variables: {missing}")
        lower = np.array([variables[name]["min"] for name in order], dtype=float)
        upper = np.array([variables[name]["max"] for name in order], dtype=float)
        return cls(order=order, lower=lower, upper=upper)

    @property
    def n_dims(self) -> int:
        return len(self.order)

    def to_dict(self, x: np.ndarray) -> dict[str, float]:
        if len(x) != self.n_dims:
            raise ValueError(f"expected {self.n_dims} values, got {len(x)}")
        return {name: float(v) for name, v in zip(self.order, x)}

    def to_array(self, design: dict[str, float]) -> np.ndarray:
        return np.array([design[name] for name in self.order], dtype=float)

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lower, self.upper)

    def sample_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Independent uniform samples, shape (n, n_dims). Used for EA population init."""
        return rng.uniform(self.lower, self.upper, size=(n, self.n_dims))

    def sample_lhs(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Latin hypercube samples, shape (n, n_dims). Used to build the surrogate training set
        so every design variable's range is evenly covered even for small sample counts."""
        cut = np.linspace(0.0, 1.0, n + 1)
        u = rng.uniform(size=(n, self.n_dims))
        a, b = cut[:n], cut[1 : n + 1]
        points = np.empty((n, self.n_dims))
        for j in range(self.n_dims):
            perm = rng.permutation(n)
            points[:, j] = a[perm] + u[:, j] * (b[perm] - a[perm])
        return self.lower + points * (self.upper - self.lower)
