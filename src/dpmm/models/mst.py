# A generative model training algorithm based on
# "Winning the NIST Contest: A scalable and general approach to differentially private synthetic data"
# by Ryan McKenna, Gerome Miklau, Daniel Sheldon
# Adapted from: https://github.com/ryan112358/private-pgm/blob/1da21c8b38149b05f1385b8e54116568b700b4fa/mechanisms/mst.py


# import argparse
import itertools
from logging import getLogger
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Self

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from disjoint_set import DisjointSet
from numpy.random import RandomState
from scipy import sparse
from scipy.special import logsumexp

from dpmm.models.base.compression import DomainCompressor
from dpmm.models.base.graphical import GraphicalGenerativeModel
from dpmm.models.base.mbi import Dataset, Domain, FactoredInference, GraphicalModel
from dpmm.models.base.mechanisms import cdp_rho
from dpmm.models.base.memory import clique_size, model_size
from dpmm.models.base.utils import gaussian_noise
from dpmm.utils import to_path
from dpmm.models.base.mechanisms import Mechanism

"""
This is a generalization of the winning mechanism from the
2018 NIST Differential Privacy Synthetic Data Competition.

Unlike the original implementation, this one can work for any discrete dataset,
and does not rely on public provisional data for measurement selection.
"""


logger = getLogger("dpgm")


def compute_weight(est, data, clique):
    a, b = clique
    model_size = clique_size(data, (a, b))
    xhat = est.project([a, b]).datavector()
    x = data.project([a, b]).datavector()
    weight = np.linalg.norm(x - xhat, 1)
    return a, b, weight, model_size


class MST(Mechanism):
    """
    NB: MST works on binned data
    """

    def __init__(
        self,
        epsilon=None,
        delta=None,
        n_iters=10000,
        compress=True,
        domain=None,
        prng: RandomState = None,
        max_model_size: int = None,
        structural_zeros=None,
        n_jobs: int = -1,
    ):

        super().__init__(
            epsilon=epsilon,
            delta=delta,
            prng=prng,
            max_model_size=max_model_size,
            compress=compress,
            domain=domain,
            structural_zeros=structural_zeros,
            n_jobs=n_jobs,
        )

        self.rho = cdp_rho(self.epsilon, self.delta)
        self.sigma = np.sqrt(3 / (2 * self.rho))
        self.n_iters = n_iters

    def _fit(self, data: Dataset, public=False):

        # select all 1-way marginals
        self.cliques = cliques_1 = [(col,) for col in data.domain]

        log1 = self.measure(data, cliques=cliques_1, public=public)
        # comress domain of all 1-way marginals
        if self.compress:
            log1 = self.compressor.fit(log1)
            data = self.compressor.transform(data)

        self.model_size = model_size(data, cliques_1)
        # select higher marginals
        if len(data.domain.attrs) > 1:
            cliques_2 = self.select(data, self.rho / 3.0, log1)
            # measure higher marginals
            log2 = self.measure(data, cliques=cliques_2, flatten=True)
            self.cliques += cliques_2
            # TODO document
        else:
            log2 = []

        return data, log1 + log2

    def select(self, data, rho, measurement_log, cliques=[], public=False):
        engine = FactoredInference(data.domain, iters=2500)
        est = engine.estimate(measurement_log)

        weights = {}
        candidates = list(itertools.combinations(data.domain.attrs, 2))
        for a, b, weight, model_size in Pool(self.n_jobs).starmap(
            compute_weight,
            zip(itertools.cycle([est]), itertools.cycle([data]), candidates),
        ):
            weights[a, b] = weight
            self.model_size += model_size
            if self.max_model_size is not None:
                if self.model_size > self.max_model_size:
                    break

        T = nx.Graph()
        T.add_nodes_from(data.domain.attrs)
        ds = DisjointSet()

        for e in cliques:
            T.add_edge(*e)
            ds.union(*e)

        r = len(list(nx.connected_components(T)))
        epsilon = np.sqrt(8 * rho / (r - 1))
        for i in range(r - 1):
            candidates = [e for e in candidates if not ds.connected(*e)]
            wgts = np.array([weights[e] for e in candidates])
            if public:
                idx = np.argmax(wgts)
            else:
                idx = self.exponential_mechanism(wgts, epsilon, sensitivity=1.0)

            e = candidates[idx]
            T.add_edge(*e)
            ds.union(*e)

        return list(T.edges)

    def exponential_mechanism(self, q, eps, sensitivity, monotonic=False):
        # TODO move to privacy utils (does not need self)
        coef = 1.0 if monotonic else 0.5
        scores = coef * eps / sensitivity * q
        probas = np.exp(scores - logsumexp(scores))
        return self.prng.choice(q.size, p=probas)

    def store(self, path: Path):
        joblib.dump(
            {
                "epsilon": self.epsilon,
                "delta": self.delta,
                "compress": self._compress,
                "compressor": self.compressor,
                "cliques": self.cliques,
                "rho": self.rho,
                "sigma": self.sigma,
                "n_iters": self.n_iters,
                "_domain": self._domain,
                "max_model_size": self.max_model_size,
                "model_size": self.model_size,
                "fit_state": self.fit_state,
            },
            path / "state.joblib",
        )

        if self.model is not None:
            self.model.save(path / "estimator.pickle")

    @classmethod
    def load(cls, path: Path) -> Self:
        state = joblib.load(path / "state.joblib")
        obj = cls(
            epsilon=state["epsilon"],
            delta=state["delta"],
            n_iters=state["n_iters"],
            compress=state["compress"],
            domain=state["_domain"],
            max_model_size=state["max_model_size"],
        )
        obj.cliques = state["cliques"]
        obj.fit_state = state["fit_state"]
        obj.compressor = state["compressor"]
        obj.model_size = state["model_size"]
        model_path = path / "estimator.pickle"
        if model_path.exists():
            obj.model = GraphicalModel.load(model_path)
        return obj


class MSTGM(GraphicalGenerativeModel):
    name = "mst"

    def __init__(
        self,
        epsilon=1,
        delta=1e-5,
        n_iters=5000,
        compress=True,
        max_model_size: int = None,
        domain=None,
        random_state: RandomState = None,
    ):
        super().__init__(domain=domain, random_state=random_state)

        self.epsilon = epsilon
        self.delta = delta

        self.generator = MST(
            epsilon=self.epsilon,
            delta=self.delta,
            n_iters=n_iters,
            compress=compress,
            domain=domain,
            prng=random_state,
            max_model_size=max_model_size,
        )

    def set_random_state(self, random_state: RandomState):
        super().set_random_state(random_state)
        self.generator.set_random_state(random_state)

    def set_domain(self, domain: Dict):
        super().set_domain(domain=domain)
        self.generator.set_domain(domain=domain)

    def fit(self, df, public=False):
        self.generator.fit(df, public=public)

    def generate(self, n_records: int = None, condition_records: pd.DataFrame = None):
        return self.generator.generate(
            n_records=n_records, condition_records=condition_records
        )

    @to_path
    def store(self, path: Path):
        super().store(path)
        self.generator.store(path)

    @classmethod
    @to_path
    def load(cls, path: Path) -> Self:
        generator = MST.load(path)
        obj = cls(epsilon=generator.epsilon, delta=generator.delta)
        del obj.generator
        obj.generator = generator

        return obj
