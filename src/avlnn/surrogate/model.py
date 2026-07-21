"""PyTorch MLP surrogate: design vector -> AVL-derived output vector (dataset.OUTPUT_COLUMNS).

Trained on the dataset from surrogate/dataset.py so the EA (surrogate_ea.py) can evaluate an
entire population in milliseconds instead of running AVL for every individual.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from avlnn.design_space import DesignSpace
from avlnn.surrogate.dataset import OUTPUT_COLUMNS


@dataclasses.dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class SurrogateMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: tuple[int, ...] = (128, 128, 64)):
        super().__init__()
        self.hidden = hidden
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclasses.dataclass
class SurrogateBundle:
    model: SurrogateMLP
    x_normalizer: Normalizer
    y_normalizer: Normalizer
    input_order: tuple[str, ...]
    output_order: tuple[str, ...]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: (n, n_in) raw design vectors -> (n, n_out) raw predicted outputs."""
        self.model.eval()
        with torch.no_grad():
            xn = self.x_normalizer.transform(x.astype(np.float32))
            yn = self.model(torch.as_tensor(xn, dtype=torch.float32)).numpy()
        return self.y_normalizer.inverse(yn)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hidden": self.model.hidden,
                "x_mean": self.x_normalizer.mean, "x_std": self.x_normalizer.std,
                "y_mean": self.y_normalizer.mean, "y_std": self.y_normalizer.std,
                "input_order": self.input_order, "output_order": self.output_order,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "SurrogateBundle":
        ckpt = torch.load(path, weights_only=False)
        model = SurrogateMLP(
            n_in=len(ckpt["input_order"]), n_out=len(ckpt["output_order"]),
            hidden=tuple(ckpt["hidden"]),
        )
        model.load_state_dict(ckpt["state_dict"])
        return cls(
            model=model,
            x_normalizer=Normalizer(ckpt["x_mean"], ckpt["x_std"]),
            y_normalizer=Normalizer(ckpt["y_mean"], ckpt["y_std"]),
            input_order=tuple(ckpt["input_order"]), output_order=tuple(ckpt["output_order"]),
        )


def train_surrogate(
    df: pd.DataFrame,
    space: DesignSpace,
    epochs: int = 300,
    lr: float = 1e-3,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[SurrogateBundle, dict[str, float]]:
    rng = np.random.default_rng(seed)
    x = df[list(space.order)].to_numpy(dtype=np.float32)
    y = df[OUTPUT_COLUMNS].to_numpy(dtype=np.float32)

    n = len(df)
    idx = rng.permutation(n)
    n_val = max(1, round(val_fraction * n))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_norm = Normalizer.fit(x[train_idx])
    y_norm = Normalizer.fit(y[train_idx])

    model = SurrogateMLP(n_in=x.shape[1], n_out=y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x_train = torch.as_tensor(x_norm.transform(x[train_idx]), dtype=torch.float32)
    y_train = torch.as_tensor(y_norm.transform(y[train_idx]), dtype=torch.float32)
    x_val = torch.as_tensor(x_norm.transform(x[val_idx]), dtype=torch.float32)
    y_val = torch.as_tensor(y_norm.transform(y[val_idx]), dtype=torch.float32)

    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(x_val), y_val).item()

    bundle = SurrogateBundle(
        model=model, x_normalizer=x_norm, y_normalizer=y_norm,
        input_order=tuple(space.order), output_order=tuple(OUTPUT_COLUMNS),
    )
    return bundle, {"val_mse": val_loss, "n_train": float(len(train_idx)), "n_val": float(len(val_idx))}
