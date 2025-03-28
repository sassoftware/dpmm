import pytest
import numpy as np
import pandas as pd
from dpmm.processing.binners import PrivTreeBinner, UniformBinner


N_RECORDS = 10_000
PT_MULTIPLIER = 2


def confirm_is_int(samples):
    # Confirm that the samples are of integer type
    assert pd.Series(samples).dtype.kind in "ui"
    return samples


def get_samples(dkind="f", size=10_000):
    # Generate samples of different data kinds
    if dkind in "ui":  # Integer
        lower = -10
        upper = 10
        samples = np.random.randint(low=lower, high=upper, size=size)
    elif dkind == "f":  # Float
        lower = -1
        upper = 1
        samples = np.random.uniform(low=lower, high=upper, size=size)
    elif dkind in "mM":  # datetime
        samples = np.random.uniform(low=0, high=1, size=size)
        if dkind == "M":
            lower = np.datetime64("1993-02-25")
            upper = np.datetime64("2025-02-25")
        else:
            seconds_in_day = 24 * 3600
            lower = np.timedelta64(-seconds_in_day, "s")
            upper = np.timedelta64(seconds_in_day, "s")

        samples = lower + (upper - lower) * samples

    return samples, lower, upper


def run_binner(cls, samples, n_bins=10, epsilon=None, lower=None, upper=None, zeros=None):
    # Run the binner on the samples and perform various checks
    binner = cls(n_bins=n_bins, epsilon=epsilon, lower=lower, upper=upper)

    if zeros is not None:
        binner.set_structural_zeros(zeros)

    t_samples = binner.fit_transform(samples)

    t_samples = confirm_is_int(t_samples)
    assert (t_samples >= 0).all()
    if isinstance(n_bins, int):
        assert (t_samples < n_bins).all()

    if epsilon is not None:
        if not isinstance(binner, PrivTreeBinner) and binner.has_bounds:
            assert binner.spent_epsilon is None
        else:
            assert (binner.spent_epsilon - epsilon) < 1e-3

    i_samples = pd.Series(binner.inverse_transform(t_samples))


    for bin_idx, bin_samples in i_samples.groupby(t_samples):
        bin_min, bin_max = binner.bin_bounds(bin_idx)
        if i_samples.dtype.kind not in  "ui":    
            assert (bin_samples >= bin_min).all()
            assert (bin_samples <= bin_max).all()
        else:
            assert (bin_samples - bin_min).min() >= -1
            assert (bin_samples - bin_max).max() <= 1
        

    if lower is not None and upper is not None:
        assert i_samples.between(left=lower, right=upper, inclusive="both").all()

    if zeros is not None:
        nz_series = i_samples.loc[~(pd.Series(t_samples).isin(binner.zeros))]

        for z in zeros:
            if i_samples.dtype.kind in "ui":
                z = [int(_z) for _z in z]
                
            not_zero = nz_series.between(left=z[0], right=z[1], inclusive="neither")  

            assert not_zero.sum() == 0
        


@pytest.mark.parametrize("n_bins", ["auto", 10, 20, 100])
@pytest.mark.parametrize("epsilon", [None, 10_000])
@pytest.mark.parametrize("binner_cls", [PrivTreeBinner, UniformBinner])
@pytest.mark.parametrize("dkind", list("fuiMm"))
@pytest.mark.parametrize("with_bounds", [True, False])
@pytest.mark.parametrize("with_zeros", [True, False])
def test_binner(n_bins, epsilon, binner_cls, dkind, with_bounds, with_zeros):
    # Test the binner with different parameters
    np.random.seed(2023)
    samples, lower, upper = get_samples(dkind=dkind)

    if with_zeros:
        zeros = sorted(np.random.uniform(0, 1, size=4) * (upper - lower)  + lower)
        zeros = [(z, zeros[idx + 1]) for idx, z in list(enumerate(zeros))[::2]]
    else:
        zeros = None

    if not with_bounds:
        lower, upper = None, None
    
    run_binner(
        cls=binner_cls,
        samples=samples,
        n_bins=n_bins,
        epsilon=epsilon,
        lower=lower,
        upper=upper,
        zeros=zeros,
    )


@pytest.mark.parametrize("n_bins", ["auto", 10, 20, 100])
@pytest.mark.parametrize("epsilon", [None])
@pytest.mark.parametrize("binner_cls", [UniformBinner, PrivTreeBinner])
@pytest.mark.parametrize("dkind", list("fuiMm"))
@pytest.mark.parametrize("rnd_seed", [42, 123])
def test_rng_parameter(n_bins, epsilon, binner_cls, dkind, rnd_seed):
    np.random.seed(2023)
    samples, lower, upper = get_samples(dkind=dkind)

    binner_1 = binner_cls(n_bins=n_bins, epsilon=epsilon, lower=lower, upper=upper)
    binner_1.set_random_state(np.random.RandomState(rnd_seed))
    t_samples_1 = binner_1.fit_transform(samples)
    i_samples_1 = binner_1.inverse_transform(t_samples_1)

    binner_2 = binner_cls(n_bins=n_bins, epsilon=epsilon, lower=lower, upper=upper)
    binner_2.set_random_state(np.random.RandomState(rnd_seed))
    t_samples_2 = binner_2.fit_transform(samples)
    i_samples_2 = binner_2.inverse_transform(t_samples_2)


    t_diff = t_samples_1 != t_samples_2

    assert np.array_equal(t_samples_1, t_samples_2)
    assert np.array_equal(i_samples_1, i_samples_2)

    binner = binner_cls(n_bins=n_bins, epsilon=epsilon, lower=lower, upper=upper)
    binner.set_random_state(np.random.RandomState(rnd_seed + 1))
    t_samples_3 = binner.fit_transform(samples)
    i_samples_3 = binner.inverse_transform(t_samples_3)

    assert not np.array_equal(i_samples_1, i_samples_3)
