"""
Code from :
- https://github.com/opendp/smartnoise-sdk/blob/10b152700749aa7dbe85f73d6382e6899ea256f6/synth/snsynth/transform/mechanism.py
- https://desfontain.es/thesis/Usability.html

"""

import math

import numpy as np
import pandas as pd
from opendp.domains import atom_domain
from opendp.measurements import make_laplace
from opendp.metrics import absolute_distance
from opendp.mod import enable_features
from typing import List, Optional, Tuple, Union


STD_DT = pd.to_datetime("01.01.1990").to_numpy()
STD_TD = pd.to_timedelta("1 hour").to_numpy()


def approx_bounds(
    vals: Union[List[float], np.ndarray], epsilon: float
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate the minimum and maximum values of a list of values.
    from: https://desfontain.es/thesis/Usability.html#usability-u-ding-

    :param vals: A list of values. Must be numeric.
    :param epsilon: The privacy budget to spend estimating the bounds.
    :return: A tuple of the estimated minimum and maximum values.

    .. code-block:: python

        vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        lower, upper = approx_bounds(vals, 0.1)
    """

    def get_bins(n_bins):
        return (
            list(-1 * 2 ** np.arange(n_bins)[::-1]) + [0] + list(2 ** np.arange(n_bins))
        )

    n_bins = 32
    vals = np.array(vals)

    dkind = vals.dtype.kind
    if dkind == "M":
        # Standardise to float
        vals = (vals - STD_DT) / STD_TD
    elif dkind == "m":
        # Standardise to float
        vals = vals / STD_TD

    bins = get_bins(n_bins)
    hist, _ = np.histogram(vals[(vals >= bins[0]) & (vals < bins[-1])], bins=bins)

    def edges(idx):
        return bins[idx], bins[idx + 1]

    # add noise
    enable_features("floating-point", "contrib")

    discovered_scale = 1.0 / epsilon
    input_space = atom_domain(T=float), absolute_distance(T=float)
    meas = make_laplace(*input_space, scale=discovered_scale)
    hist = [meas(float(v)) for v in hist]
    n_bins = len(hist)

    failure_prob = 10e-9
    highest_failure_prob = 1 / (n_bins * 2)

    exceeds = []
    while len(exceeds) < 1 and failure_prob <= highest_failure_prob:
        p = 1 - failure_prob
        K = -np.log(2 - 2 * p ** (1 / (n_bins - 1))) / epsilon
        exceeds = [idx for idx, v in enumerate(hist) if v > K]
        failure_prob *= 10

    if len(exceeds) == 0:
        return (None, None)

    lower, upper = min(exceeds), max(exceeds)
    ll, _ = edges(lower)
    _, uu = edges(upper)

    if dkind == "M":
        ll, uu = pd.Timestamp(STD_DT + ll * STD_TD), pd.Timestamp(STD_DT + uu * STD_TD)
    elif dkind == "m":
        ll, uu = pd.Timedelta(ll * STD_TD), pd.Timedelta(uu * STD_TD)
    elif dkind == "f":
        ll, uu = float(ll), float(uu)
    return (ll, uu)


def optimal_n_bins(
    series: pd.Series, upper: Optional[int] = None, epsilon: Optional[float] = None
) -> int:
    """Calculate the optimal number of bins for a histogram.

    :param series: A pandas Series containing the data.
    :param upper: An optional upper limit for the number of bins.
    :param epsilon: An optional privacy budget parameter.
    :return: The optimal number of bins as an integer.

    .. code-block:: python

        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        n_bins = optimal_n_bins(series, upper=20, epsilon=0.1)
    """
    n_series = len(series)
    n_bins = 2 * math.pow(n_series, 1 / 3)

    if pd.notnull(epsilon):
        n_bins *= 1 / (1 + np.exp(-epsilon))  # ].5 - 1[
        n_bins = max(n_bins, 5)

    if upper is not None:
        n_bins = min(n_bins, upper)
    return int(np.ceil(n_bins))
