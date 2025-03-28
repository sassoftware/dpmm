import itertools
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Self

import joblib
import numpy as np
import pandas as pd
from numpy.random import RandomState
from scipy import sparse

from dpmm.models.base.compression import DomainCompressor
from dpmm.models.base.graphical import GraphicalGenerativeModel
from dpmm.models.base.matrix import Identity
from dpmm.models.base.mbi import Dataset, Domain, FactoredInference, GraphicalModel
from dpmm.models.base.mechanisms import Mechanism
from dpmm.models.base.utils import gaussian_noise
from dpmm.utils import to_path
from scipy.special import softmax


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def hypothetical_model_size(domain, cliques):
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2**20


def compile_workload(workload):
    def score(cl):
        return sum(len(set(cl) & set(ax)) for ax in workload)

    return {cl: score(cl) for cl in downward_closure(workload)}


def filter_candidates(candidates, model, size_limit):
    ans = {}
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        if size_limit is None:
            cond1 = True
        else:
            cond1 = (
                hypothetical_model_size(model.domain, model.cliques + [cl])
                <= size_limit
            )
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans


def measure_one_way(cl, data, sigma, prng, public=False):
    # TODO: figure out to make determinstic
    x = data.project(cl).datavector()
    if public:
        y = x
    else:
        y = x + gaussian_noise(sigma=sigma, size=x.size)
    I = Identity(y.size)
    return (I, y, sigma, cl)


def _measure(data, proj, wgt, sigma, public=False):
    # TODO: figure out to make determinstic
    x = data.project(proj).datavector()
    if public:
        y = x
    else:
        y = x + gaussian_noise(sigma=sigma / wgt, size=x.size)
    Q = sparse.eye(x.size)
    return (Q, y, sigma / wgt, proj)


def measure(data, cliques, sigma, weights=None, public=False, n_jobs=-1):
    if weights is None:
        weights = np.ones(len(cliques))
    weights = np.array(weights) / np.linalg.norm(weights)
    measurements = [
        meas
        for meas in Pool(n_jobs).starmap(
            _measure,
            zip(
                itertools.cycle([data]),
                cliques,
                weights,
                itertools.cycle([sigma]),
                itertools.cycle([public]),
            ),
        )
    ]
    return measurements


class AIM(Mechanism):
    def __init__(
        self,
        epsilon=1,
        delta=1e-5,
        prng: RandomState = None,
        rounds=None,
        max_model_size=80,
        n_iters: int = 1000,
        degree=2,
        num_marginals=None,
        max_cells=10000,
        structural_zeros={},
        compress=False,
        domain=None,
        n_jobs=-1,
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

        self.rounds = rounds
        self.degree = degree
        self.n_iters = n_iters
        self.num_marginals = num_marginals
        self.max_cells = max_cells

    def _worst_approximated(self, cl, wgt, x, model, sigma):
        bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
        xest = model.project(cl).datavector()
        error = wgt * (np.linalg.norm(x - xest, 1) - bias)
        sensitivity = abs(wgt)
        return cl, error, sensitivity

    def exponential_mechanism(
        self, qualities, epsilon, sensitivity=1.0, base_measure=None
    ):
        if isinstance(qualities, dict):
            keys = list(qualities.keys())
            qualities = np.array([qualities[key] for key in keys])
            if base_measure is not None:
                base_measure = np.log([base_measure[key] for key in keys])
        else:
            qualities = np.array(qualities)
            keys = np.arange(qualities.size)

        """ Sample a candidate from the permute-and-flip mechanism """
        q = qualities - qualities.max()
        if base_measure is None:
            p = softmax(0.5 * epsilon / sensitivity * q)
        else:
            p = softmax(0.5 * epsilon / sensitivity * q + base_measure)

        return keys[self.prng.choice(p.size, p=p)]

    def worst_approximated(self, candidates, answers, model, eps, sigma, public=False):
        errors = {}
        sensitivity = {}

        for cl, err, sens in Pool(self.n_jobs).starmap(
            self._worst_approximated,
            [(cl, candidates[cl], answers[cl], model, sigma) for cl in candidates],
        ):
            errors[cl] = err
            sensitivity[cl] = sens

        max_sensitivity = max(
            sensitivity.values()
        )  # if all weights are 0, could be a problem

        if public:
            _cliques = list(errors.keys())
            worst = np.argmax(errors.values())
            return _cliques[worst]
        else:
            return self.exponential_mechanism(errors, eps, max_sensitivity)

    def _fit(self, data: Dataset, public=False, W=None):

        if W is None:
            W = list(itertools.combinations(data.domain, self.degree))
            W = [cl for cl in W if data.domain.size(cl) <= self.max_cells]
            W = [(cl, 1.0) for cl in W]

        if self.num_marginals is not None:
            W = [
                W[i]
                for i in self.prng.choice(len(W), self.num_marginals, replace=False)
            ]

        rounds = self.rounds or 16 * len(data.domain)
        workload = [cl for cl, _ in W]
        candidates = compile_workload(workload)

        oneway = [cl for cl in candidates if len(cl) == 1]

        sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
        epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)

        measurements = []

        # cliques & sigmas
        self.cliques = list(oneway)

        rho_used = len(oneway) * 0.5 / sigma**2
        if self.n_jobs > 1:
            with Pool(self.n_jobs) as p:
                oneway_measurements = p.starmap(
                    measure_one_way,
                    [(cl, data, sigma, self.prng, public) for cl in oneway],
                )
            measurements.extend(oneway_measurements)
        else:
            measurements = [
                measure_one_way(cl, data.project(cl), sigma, self.prng, public)
                for cl in oneway
            ]

        if self.compress:
            measurements = self.compressor.fit(measurements)
            data = self.compressor.transform(data)

        self.engine = FactoredInference(
            data.domain,
            iters=self.n_iters,
            warm_start=True,
            prng=self.prng,
        )

        answers = {cl: data.project(cl).datavector() for cl in candidates}

        model = self.engine.estimate(measurements)

        t = 0
        terminate = False
        while not terminate:
            t += 1
            if public and t >= rounds:
                break
            elif self.rho - rho_used < 2 * (0.5 / sigma**2 + 1.0 / 8 * epsilon**2):
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                if remaining < 0:
                    break
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True

            rho_used += 1.0 / 8 * epsilon**2 + 0.5 / sigma**2
            if self.max_model_size is not None:
                size_limit = self.max_model_size * rho_used / self.rho
            else:
                size_limit = None

            small_candidates = filter_candidates(candidates, model, size_limit)
            cl = self.worst_approximated(
                small_candidates, answers, model, epsilon, sigma, public
            )

            n = data.domain.size(cl)
            Q = Identity(n)
            x = data.project(cl).datavector()
            # TODO: figure out to make determinstic
            if public:
                y = x
            else:
                y = x + gaussian_noise(sigma=sigma, size=n)

            self.cliques.append(cl)
            measurements.append((Q, y, sigma, cl))
            z = model.project(cl).datavector()

            model = self.engine.estimate(measurements)
            w = model.project(cl).datavector()
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                sigma /= 2
                epsilon *= 2

        return data, measurements

    def store(self, path: Path):
        joblib.dump(
            {
                "epsilon": self.epsilon,
                "delta": self.delta,
                "rounds": self.rounds,
                "n_iters": self.n_iters,
                "max_model_size": self.max_model_size,
                "degree": self.degree,
                "num_marginals": self.num_marginals,
                "max_cells": self.max_cells,
                "structural_zeros": self.structural_zeros,
                "_domain": self._domain,
                "compress": self._compress,
                "compressor": self.compressor,
                "cliques": self.cliques,
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
            rounds=state["rounds"],
            n_iters=state["n_iters"],
            max_model_size=state["max_model_size"],
            degree=state["degree"],
            num_marginals=state["num_marginals"],
            max_cells=state["max_cells"],
            structural_zeros=state["structural_zeros"],
            domain=state["_domain"],
            compress=state["compress"],
        )
        obj.fit_state = state["fit_state"]
        obj.cliques = state["cliques"]
        obj.compressor = state["compressor"]
        model_path = path / "estimator.pickle"
        if model_path.exists():
            obj.model = GraphicalModel.load(model_path)
        return obj


class AIMGM(GraphicalGenerativeModel):
    name = "aim"

    def __init__(
        self,
        epsilon=1,
        delta=1e-5,
        rounds=None,
        compress=True,
        n_iters: int = 1000,
        max_model_size=80,
        degree=2,
        num_marginals=None,
        max_cells=10000,
        structural_zeros={},
        domain=None,
        random_state: RandomState = None,
    ):

        super().__init__(domain=domain, random_state=random_state)
        self.epsilon = epsilon
        self.delta = delta

        self.generator = AIM(
            epsilon=epsilon,
            delta=delta,
            prng=random_state,
            n_iters=n_iters,
            rounds=rounds,
            compress=compress,
            max_model_size=max_model_size,
            degree=degree,
            num_marginals=num_marginals,
            max_cells=max_cells,
            structural_zeros=structural_zeros,
            domain=domain,
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
        generator = AIM.load(path)
        obj = cls(
            epsilon=generator.epsilon,
            delta=generator.delta,
            rounds=generator.rounds,
            max_model_size=generator.max_model_size,
            degree=generator.degree,
            num_marginals=generator.num_marginals,
            max_cells=generator.max_cells,
            structural_zeros=generator.structural_zeros,
            domain=generator._domain,
            compress=generator.compress,
        )
        del obj.generator
        obj.generator = generator

        return obj
