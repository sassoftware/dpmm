import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# convert mu-GDP to (eps, delta)-DP using Equation (6) from Tight Auditing DPML paper
def delta_from_eps_mu(eps, mu):
    return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def mu_from_eps_delta(eps, delta):
    # bracket search
    lo, hi = 1e-6, 50.0

    # expand hi if needed
    while delta_from_eps_mu(eps, hi) < delta:
        hi *= 2
        if hi > 1e6:
            raise RuntimeError("Failed to bracket μ")

    return brentq(lambda m: delta_from_eps_mu(eps, m) - delta, lo, hi)
