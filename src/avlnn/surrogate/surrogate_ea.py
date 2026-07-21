"""EA driven by the NN surrogate for speed, with periodic real-AVL re-validation of the
current elite and active-learning retraining -- so surrogate accuracy keeps improving exactly
where it matters most: near the constraint boundaries the EA is pushing the population toward.

The evolved population carries over from block to block (via run_ea's initial_population);
only the fitness function changes as the surrogate retrains, so the search is one continuous
EA whose evaluator gets progressively more accurate.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
import pandas as pd

from avlnn.config import Constants
from avlnn.design_space import DesignSpace
from avlnn.ea import EaConfig, EaResult, GenerationRecord, run_ea
from avlnn.evaluate import evaluate_design
from avlnn.objective import ObjectiveResult
from avlnn.surrogate.dataset import result_to_row
from avlnn.surrogate.model import SurrogateBundle, train_surrogate

# Called with (design_array, real_avl_result) for every elite re-validated against real AVL.
OnRevalidateFn = Callable[[np.ndarray, ObjectiveResult], None]


@dataclasses.dataclass
class SurrogateEaConfig:
    ea_config: EaConfig
    revalidate_every_n_generations: int = 5
    n_elite_to_revalidate: int = 5
    retrain_epochs: int = 100


def _surrogate_fitness_fn(bundle: SurrogateBundle):
    fitness_idx = bundle.output_order.index("fitness")

    def fn(x: np.ndarray) -> float:
        pred = bundle.predict(x[None, :])[0]
        return float(pred[fitness_idx])

    return fn


def run_surrogate_assisted_ea(
    space: DesignSpace,
    constants: Constants,
    initial_bundle: SurrogateBundle,
    initial_dataset: pd.DataFrame,
    config: SurrogateEaConfig,
    on_generation: Callable[[GenerationRecord], None] | None = None,
    on_revalidate: OnRevalidateFn | None = None,
) -> tuple[EaResult, SurrogateBundle, pd.DataFrame]:
    """Runs the EA in blocks of `revalidate_every_n_generations`. After each block, the
    block's top-K individuals are re-evaluated with real AVL (reported via `on_revalidate`),
    folded into the training set, and the surrogate is retrained before the next block.
    `on_generation` receives records with cumulative generation numbers across blocks;
    its best_fitness values are surrogate predictions, not AVL truth."""
    bundle = initial_bundle
    dataset = initial_dataset
    result: EaResult | None = None
    population: np.ndarray | None = None
    remaining = config.ea_config.n_generations
    gen_offset = 0
    block_index = 0

    while remaining > 0:
        block_generations = min(config.revalidate_every_n_generations, remaining)
        remaining -= block_generations
        block_seed = (
            None if config.ea_config.seed is None else config.ea_config.seed + block_index
        )
        block_config = dataclasses.replace(
            config.ea_config, n_generations=block_generations, seed=block_seed,
        )

        def block_on_generation(record: GenerationRecord, _offset: int = gen_offset) -> None:
            if on_generation is not None:
                on_generation(dataclasses.replace(record, generation=record.generation + _offset))

        result = run_ea(
            space, _surrogate_fitness_fn(bundle), block_config,
            on_generation=block_on_generation, initial_population=population,
        )
        population = result.final_population
        gen_offset += block_generations
        block_index += 1

        new_rows = []
        for x in result.top_k(config.n_elite_to_revalidate):
            design = space.to_dict(x)
            real_result = evaluate_design(design, constants)
            if on_revalidate is not None:
                on_revalidate(x, real_result)
            row = dict(zip(space.order, x))
            row.update(result_to_row(real_result))
            new_rows.append(row)

        dataset = pd.concat([dataset, pd.DataFrame(new_rows)], ignore_index=True)
        bundle, _metrics = train_surrogate(dataset, space, epochs=config.retrain_epochs)

    assert result is not None, "n_generations must be > 0"
    return result, bundle, dataset
