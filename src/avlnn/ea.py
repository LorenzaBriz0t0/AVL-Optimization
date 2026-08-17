"""A real-valued evolutionary algorithm: tournament selection, BLX-alpha crossover, Gaussian
mutation, elitism.

Deliberately decoupled from AVL -- it only needs an `evaluate_fn(x: np.ndarray) -> float`
fitness callable, so this exact engine drives both the pure-AVL search (scripts/run_ea.py) and
the NN-surrogate search (surrogate/surrogate_ea.py) without duplication.
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
from typing import Callable

import numpy as np

from avlnn.design_space import DesignSpace

EvaluateFn = Callable[[np.ndarray], float]


@dataclasses.dataclass
class EaConfig:
    population_size: int = 40
    n_generations: int = 50
    elite_fraction: float = 0.1
    tournament_size: int = 3
    crossover_prob: float = 0.8
    blx_alpha: float = 0.3
    mutation_prob: float = 0.15
    mutation_sigma_frac: float = 0.1   # mutation stdev as a fraction of each variable's range
    n_workers: int = 1
    seed: int | None = None


@dataclasses.dataclass
class GenerationRecord:
    generation: int
    best_fitness: float
    design: np.ndarray


@dataclasses.dataclass
class EaResult:
    best_design: np.ndarray
    best_fitness: float
    history: list[GenerationRecord]
    final_population: np.ndarray
    final_fitness: np.ndarray

    def top_k(self, k: int) -> np.ndarray:
        """The k distinct-fittest individuals from the final population (for re-validation)."""
        order = np.argsort(self.final_fitness)[::-1][:k]
        return self.final_population[order]


def _evaluate_population(pop: np.ndarray, evaluate_fn: EvaluateFn, n_workers: int) -> np.ndarray:
    if n_workers <= 1:
        return np.array([evaluate_fn(ind) for ind in pop])
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        return np.array(list(pool.map(evaluate_fn, pop)))


def _tournament_select(
    pop: np.ndarray, fitness: np.ndarray, k: int, rng: np.random.Generator,
) -> np.ndarray:
    idx = rng.integers(0, len(pop), size=k)
    winner = idx[np.argmax(fitness[idx])]
    return pop[winner]


def _blx_alpha_crossover(
    p1: np.ndarray, p2: np.ndarray, alpha: float, rng: np.random.Generator,
) -> np.ndarray:
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    span = hi - lo
    return rng.uniform(lo - alpha * span, hi + alpha * span)


def _mutate(
    ind: np.ndarray, space: DesignSpace, prob: float, sigma_frac: float, rng: np.random.Generator,
) -> np.ndarray:
    mask = rng.random(len(ind)) < prob
    sigma = sigma_frac * (space.upper - space.lower)
    noise = rng.normal(0.0, sigma)
    return space.clip(np.where(mask, ind + noise, ind))


def run_ea(
    space: DesignSpace,
    evaluate_fn: EvaluateFn,
    config: EaConfig,
    on_generation: Callable[[GenerationRecord], None] | None = None,
    initial_population: np.ndarray | None = None,
) -> EaResult:
    """`initial_population` (shape (population_size, n_dims)) seeds the population instead of
    uniform random sampling -- used by the surrogate-assisted EA to carry the evolved
    population across surrogate-retraining blocks instead of restarting each block."""
    rng = np.random.default_rng(config.seed)
    if initial_population is not None:
        pop = space.clip(np.asarray(initial_population, dtype=float))
        expected_shape = (config.population_size, space.n_dims)
        if pop.shape != expected_shape:
            raise ValueError(f"initial_population shape {pop.shape} != {expected_shape}")
    else:
        pop = space.sample_uniform(config.population_size, rng)
    fitness = _evaluate_population(pop, evaluate_fn, config.n_workers)

    n_elite = max(1, round(config.elite_fraction * config.population_size))
    history: list[GenerationRecord] = []

    for gen in range(config.n_generations):
        elite_idx = np.argsort(fitness)[::-1][:n_elite]
        elites = pop[elite_idx]
        elite_fitness = fitness[elite_idx]

        children = []
        while len(children) < config.population_size - n_elite:
            p1 = _tournament_select(pop, fitness, config.tournament_size, rng)
            p2 = _tournament_select(pop, fitness, config.tournament_size, rng)
            child = (
                _blx_alpha_crossover(p1, p2, config.blx_alpha, rng)
                if rng.random() < config.crossover_prob
                else p1.copy()
            )
            children.append(_mutate(child, space, config.mutation_prob, config.mutation_sigma_frac, rng))

        # Elites' genomes are unchanged, and evaluate_fn (AVL or the surrogate) is
        # deterministic, so re-running them would just waste evaluations -- only the new
        # children need fresh fitness values.
        children_arr = np.array(children)
        children_fitness = _evaluate_population(children_arr, evaluate_fn, config.n_workers)
        pop = np.vstack([elites, children_arr])
        fitness = np.concatenate([elite_fitness, children_fitness])

        best_i = int(np.argmax(fitness))
        record = GenerationRecord(
            generation=gen, best_fitness=float(fitness[best_i]), design=pop[best_i].copy(),
        )
        history.append(record)
        if on_generation is not None:
            on_generation(record)

    best_i = int(np.argmax(fitness))
    return EaResult(
        best_design=pop[best_i], best_fitness=float(fitness[best_i]), history=history,
        final_population=pop, final_fitness=fitness,
    )
