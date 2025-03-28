import numpy as np
import pandas as pd

import itertools
from numpy.random import RandomState
from math import exp, sqrt
from functools import partial
from scipy import sparse
from typing import Dict
from scipy.special import erf, softmax

from multiprocessing import Pool, cpu_count
from dpmm.models.base.mbi import Dataset, Domain, FactoredInference, GraphicalModel
from dpmm.models.base.compression import DomainCompressor
from dpmm.models.base.utils import gaussian_noise
from dpmm.models.base.memory import model_size
from dpmm.models.base.mechanisms import cdp_rho


class Mechanism:
    def __init__(
        self,
        epsilon,
        delta,
        prng: RandomState = None,
        max_model_size: int = None,
        compress=False,
        domain=None,
        structural_zeros: Dict = None,
        n_jobs: int = -1,
    ):
        """
        Base class for a mechanism.
        :param epsilon: privacy parameter
        :param delta: privacy parameter
        :param prng: pseudo random number generator
        """

        # adjust for non-DP
        if not epsilon:
            epsilon = 0
            delta = 0.99999  # values >= 1 break the code

        self.epsilon = epsilon
        self.delta = delta
        self.rho = 0 if delta == 0 else cdp_rho(epsilon, delta)
        self._prng = prng

        self._compress = compress

        if structural_zeros is None:
            structural_zeros = {}

        if self._compress or len(structural_zeros) > 0:
            self.compressor = DomainCompressor(
                prng=self.prng,
                structural_zeros=structural_zeros,
                zeros_only=not(self._compress),
            )
        else:
            self.compressor = None

        self.set_structural_zeros(structural_zeros)

        self._domain = domain
        self.max_model_size = max_model_size
        self.model_size = None
        self.model = None
        self.fit_state = None
        self.cliques = None
        if n_jobs == -1:
            n_jobs = cpu_count() - 1
        self.n_jobs = n_jobs
        self._measures = None

    @property
    def compress(self):
        return self.compressor is not None

    @property
    def prng(self):
        if self._prng is None:
            return np.random.RandomState()
        return self._prng

    def set_random_state(self, random_state: RandomState):
        self._prng = random_state
        if self.compress:
            self.compressor.prng = random_state

        if self.model is not None:
            self.model.set_random_state(random_state)

    def set_structural_zeros(self, structural_zeros: Dict):
        self.structural_zeros = structural_zeros
        if len(structural_zeros) > 0:
            if self.compressor is None:
                self.compressor = DomainCompressor(
                    prng=self.prng,
                    structural_zeros=structural_zeros,
                    zeros_only=not (self.compress),
                )
            else:
                self.compressor.structural_zeros = structural_zeros
        else:
            if self._compress is False:
                self.compressor = None

    def set_domain(self, domain: Dict):
        self._domain = domain

    def fit(self, df, public=False, *args, **kwargs):
        # prepare data
        if self._domain is None:
            _domain = (df.astype(int).max(axis=0) + 1).to_dict()
            if not public:
                # TODO: Add warning
                pass
            else:
                self._domain = _domain
        else:
            _domain = self._domain

        domain = Domain(
            list(df.columns), np.array([_domain[col] for col in df.columns])
        )

        data = Dataset(df, domain)

        # select cliques
        if (self.fit_state is None) or public:
            data, measures = self._fit(data=data, public=public, *args, **kwargs)
        else:
            if self.fit_state == "pretrained":
                self.rho = cdp_rho(self.epsilon, self.delta)
                self.sigma = np.sqrt(1 / (2 * self.rho))

            if self.compress:
                data = self.compressor.transform(data)
                
            measures = self.measure(data, public=public, flatten=True)

            # Model Size
            self.model_size = model_size(data, self.cliques)


        # measures
        self._measures = measures

        if not public:
            engine = FactoredInference(data.domain, self.n_iters, prng=self.prng)
            self.model = engine.estimate(measures)

        self.fit_state = "pretrained" if public else "trained"

    @property
    def measures(self):
        return self._measures

    def _measure(self, data, proj, wgt, flatten=None, public=False):
        if flatten is None:
            flatten = (self.compress)
        x = data.project(list(proj)).datavector(flatten=flatten)
        # TODO: figure out to make determinstic
        if public:
            y = x
        else:
            y = x + gaussian_noise(sigma=self.sigma / wgt, size=x.shape)
        Q = sparse.eye(x.size)
        return (Q, y, self.sigma, proj)

    def measure(self, data, cliques=None, weights=None, public=False, flatten=None):
        if cliques is None:
            cliques = self.cliques

        if weights is None:
            weights = np.ones(len(self.cliques))

        weights = np.array(weights) / np.linalg.norm(weights)
        measurements = [
            meas
            for meas in Pool(self.n_jobs).starmap(
                self._measure,
                zip(
                    itertools.cycle([data]),
                    cliques,
                    weights,
                    itertools.cycle([flatten]),
                    itertools.cycle([public]),
                ),
            )
        ]
        return measurements

    def generate(self, n_records: int = None, condition_records: pd.DataFrame = None):
        if self.fit_state != "trained":
            raise ValueError(
                "Model has not been fully trained yet. please run a .fit call with `public` set to `False`"
            )

        if condition_records is not None:
            _conditions = condition_records.copy()

        if (self.compress) and condition_records is not None:
            condition_records = self.compressor.transform(condition_records)

        synth = self.model.synthetic_data(
            rows=n_records, condition_records=condition_records
        )
        if self.compress:
            synth = self.compressor.reverse(synth)

        synth_df = synth.df
        if condition_records is not None:
            synth_df = pd.concat(
                [_conditions, synth_df.drop(columns=_conditions.columns)], axis=1
            )

        return synth_df
