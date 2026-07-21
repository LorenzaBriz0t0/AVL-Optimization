import numpy as np
import pytest

from avlnn.design_space import DesignSpace
from avlnn.ea import EaConfig, run_ea


@pytest.fixture
def space():
    return DesignSpace.load()


def _quadratic_fitness(space):
    """Maximum (0.0) at the center of the design space."""
    center = (space.lower + space.upper) / 2.0
    span = space.upper - space.lower

    def fn(x):
        return -float(np.sum(((x - center) / span) ** 2))

    return fn


def test_run_ea_improves_over_generations(space):
    config = EaConfig(population_size=20, n_generations=15, seed=0)
    result = run_ea(space, _quadratic_fitness(space), config)
    assert result.best_fitness > result.history[0].best_fitness * 0.5  # moved toward optimum
    assert len(result.history) == 15
    assert result.final_population.shape == (20, space.n_dims)
    assert result.final_fitness.shape == (20,)


def test_run_ea_uses_initial_population(space):
    """With mutation off and identical parents, BLX crossover of a point with itself is that
    same point -- so a uniform initial population must pass through unchanged."""
    point = (space.lower + space.upper) / 2.0
    initial = np.tile(point, (10, 1))
    config = EaConfig(
        population_size=10, n_generations=3, seed=0,
        mutation_prob=0.0, crossover_prob=1.0,
    )
    result = run_ea(space, _quadratic_fitness(space), config, initial_population=initial)
    np.testing.assert_allclose(result.best_design, point)
    np.testing.assert_allclose(result.final_population, initial)


def test_run_ea_initial_population_wrong_shape_raises(space):
    config = EaConfig(population_size=10, n_generations=1, seed=0)
    bad = np.zeros((5, space.n_dims))
    with pytest.raises(ValueError):
        run_ea(space, _quadratic_fitness(space), config, initial_population=bad)


def test_top_k_returns_fittest(space):
    config = EaConfig(population_size=12, n_generations=2, seed=1)
    result = run_ea(space, _quadratic_fitness(space), config)
    fn = _quadratic_fitness(space)
    top = result.top_k(3)
    assert top.shape == (3, space.n_dims)
    top_scores = [fn(x) for x in top]
    assert top_scores[0] == pytest.approx(result.best_fitness)
    assert sorted(top_scores, reverse=True) == top_scores