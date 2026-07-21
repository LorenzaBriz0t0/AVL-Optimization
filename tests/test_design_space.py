import numpy as np
import pytest

from avlnn.design_space import DesignSpace


@pytest.fixture
def space():
    return DesignSpace.load()


def test_round_trip_dict_array(space):
    x = (space.lower + space.upper) / 2.0
    d = space.to_dict(x)
    x2 = space.to_array(d)
    np.testing.assert_allclose(x, x2)
    assert set(d.keys()) == set(space.order)


def test_sample_uniform_within_bounds(space):
    rng = np.random.default_rng(0)
    samples = space.sample_uniform(500, rng)
    assert samples.shape == (500, space.n_dims)
    assert np.all(samples >= space.lower)
    assert np.all(samples <= space.upper)


def test_sample_lhs_within_bounds_and_covers_range(space):
    rng = np.random.default_rng(0)
    samples = space.sample_lhs(200, rng)
    assert samples.shape == (200, space.n_dims)
    assert np.all(samples >= space.lower)
    assert np.all(samples <= space.upper)
    # LHS should spread samples across most of each variable's range, not cluster.
    spans = samples.max(axis=0) - samples.min(axis=0)
    full_spans = space.upper - space.lower
    assert np.all(spans > 0.8 * full_spans)


def test_clip(space):
    too_high = space.upper + 100.0
    clipped = space.clip(too_high)
    np.testing.assert_allclose(clipped, space.upper)


def test_to_dict_wrong_length_raises(space):
    with pytest.raises(ValueError):
        space.to_dict(np.zeros(space.n_dims + 1))
