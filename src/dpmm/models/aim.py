# Implementation of AIM: An Adaptive and Iterative Mechanism for DP Synthetic Data.
# Note that with the default settings, AIM can take many hours to run.  You can configure
# the runtime /utility tradeoff via the max_model_size flag.  We recommend setting it to 1.0
# for debugging, but keeping the default value of 80 for any official comparisons to this mechanism.
# Note that we assume in this file that the data has been appropriately preprocessed so that there are no large-cardinality categorical attributes.  If there are, we recommend using something like "compress_domain" from mst.py.  Since our paper evaluated already-preprocessed datastes, we did not implement that here for simplicity.

# Adapted from: https://github.com/ryan112358/private-pgm/blob/master/mechanisms/aim.py

import itertools
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Self

import joblib
import numpy as np
from numpy.random import RandomState
from scipy import sparse

from dpmm.models.base.graphical import GraphicalGenerativeModel
from dpmm.models.base.matrix import Identity
from dpmm.models.base.mbi import Dataset, Domain, FactoredInference, GraphicalModel
from dpmm.models.base.mechanisms import Mechanism
from dpmm.models.base.utils import gaussian_noise
from dpmm.utils import to_path
from scipy.special import softmax


def powerset(iterable: list) -> itertools.chain:
    """
    Generate all non-empty subsets of an iterable.

    :param iterable: The input list.
    :type iterable: list
    :return: A chain object containing all subsets.
    :rtype: itertools.chain

    **Example**::

        >>> list(powerset([1, 2, 3]))
        [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def downward_closure(workload: list) -> list:
    """
    Compute the downward closure of a set of projections.

    :param workload: A list of projections.
    :type workload: list
    :return: A sorted list of subsets in the downward closure.
    :rtype: list

    **Example**::

        >>> downward_closure([(1, 2), (2, 3)])
        [(1,), (2,), (3,), (1, 2), (2, 3)]
    """
    ans = set()
    for proj in workload:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def hypothetical_model_size(domain: Domain, cliques: list) -> float:
    """
    Calculate the hypothetical size of a graphical model.

    :param domain: The domain of the model.
    :type domain: Domain
    :param cliques: A list of cliques.
    :type cliques: list
    :return: The size of the model in megabytes.
    :rtype: float

    **Example**::

        >>> hypothetical_model_size(domain, [(1, 2), (2, 3)])
        0.015625
    """
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2**20


def compile_workload(workload: list) -> Dict[tuple, int]:
    """
    Compile a workload into a dictionary of cliques and their scores.

    :param workload: A list of workload cliques.
    :type workload: list
    :return: A dictionary mapping cliques to scores.
    :rtype: Dict[tuple, int]

    **Example**::

        >>> compile_workload([(1, 2), (2, 3)])
        {(1,): 1, (2,): 2, (3,): 1, (1, 2): 2, (2, 3): 2}
    """

    def score(cl):
        return sum(len(set(cl) & set(ax)) for ax in workload)

    return {cl: score(cl) for cl in downward_closure(workload)}


def filter_candidates(
    candidates: Dict[tuple, int], model: GraphicalModel, size_limit: float
) -> Dict[tuple, int]:
    """
    Filter candidates based on size constraints and free cliques.

    :param candidates: Candidate cliques with scores.
    :type candidates: Dict[tuple, int]
    :param model: The current graphical model.
    :type model: GraphicalModel
    :param size_limit: The maximum allowed model size.
    :type size_limit: float
    :return: Filtered candidates.
    :rtype: Dict[tuple, int]

    **Example**::

        >>> filter_candidates(candidates, model, 80)
        {(1, 2): 3, (2, 3): 2}
    """
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


def measure_one_way(
    cl: tuple, data: Dataset, sigma: float, prng: RandomState, public: bool = False
) -> tuple:
    """
    Measure a single clique with optional noise.

    :param cl: The clique to measure.
    :type cl: tuple
    :param data: The dataset.
    :type data: Dataset
    :param sigma: The noise scale.
    :type sigma: float
    :param prng: The random state.
    :type prng: RandomState
    :param public: Whether the data is public. Defaults to False.
    :type public: bool, optional
    :return: Measurement matrix, noisy data vector, noise scale, and clique.
    :rtype: tuple

    **Example**::

        >>> measure_one_way((1, 2), data, 1.0, prng)
        (<Identity>, array([0.5, 0.3]), 1.0, (1, 2))
    """
    x = data.project(cl).datavector()
    if public:
        y = x
    else:
        y = x + gaussian_noise(sigma=sigma, size=x.size)
    i_matrix = Identity(y.size)
    return (i_matrix, y, sigma, cl)


def _measure(
    data: Dataset, proj: tuple, wgt: float, sigma: float, public: bool = False
) -> tuple:
    """
    Perform a measurement for a single projection.

    :param data: The dataset.
    :type data: Dataset
    :param proj: The projection.
    :type proj: tuple
    :param wgt: The weight.
    :type wgt: float
    :param sigma: The noise scale.
    :type sigma: float
    :param public: Whether the data is public. Defaults to False.
    :type public: bool, optional
    :return: Measurement matrix, noisy data vector, noise scale, and projection.
    :rtype: tuple

    **Example**::

        >>> _measure(data, (1, 2), 0.5, 1.0)
        (<sparse.eye>, array([0.5, 0.3]), 2.0, (1, 2))
    """
    x = data.project(proj).datavector()
    if public:
        y = x
    else:
        y = x + gaussian_noise(sigma=sigma, size=x.size)
    q_matrix = sparse.eye(x.size)
    return (q_matrix, y, sigma, proj)


def measure(
    data: Dataset,
    cliques: list,
    sigma: float,
    weights: np.ndarray = None,
    public: bool = False,
    n_jobs: int = -1,
) -> list:
    """
    Measure multiple cliques in parallel.

    :param data: The dataset.
    :type data: Dataset
    :param cliques: A list of cliques.
    :type cliques: list
    :param sigma: The noise scale.
    :type sigma: float
    :param weights: Weights for the cliques. Defaults to None.
    :type weights: np.ndarray, optional
    :param public: Whether the data is public. Defaults to False.
    :type public: bool, optional
    :param n_jobs: Number of parallel jobs. Defaults to -1.
    :type n_jobs: int, optional
    :return: A list of measurements.
    :rtype: list

    **Example**::

        >>> measure(data, [(1, 2), (2, 3)], 1.0)
        [(<sparse.eye>, array([0.5, 0.3]), 1.0, (1, 2)), ...]
    """
    if weights is None:
        weights = np.ones(len(cliques))
    weights = np.array(weights) / np.linalg.norm(weights)
    if n_jobs == 1:
        measurements = [
            _measure(*args)
            for args in zip(
                itertools.cycle([data]),
                cliques,
                weights,
                itertools.cycle([sigma]),
                itertools.cycle([public]),
            )
        ]
    else:
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
    """
    Adaptive and Iterative Mechanism (AIM) for differential privacy.
    The model works by iteratively selecting the worst approximated clique
    and adding noise to it.
    It uses the exponential mechanism to select the worst approximated clique
    based on the quality scores of the candidates.

    Ref: https://arxiv.org/pdf/2201.12677


    :param epsilon: Privacy budget.
    :type epsilon: float
    :param delta: Privacy parameter.
    :type delta: float
    :param prng: Random state for reproducibility.
    :type prng: RandomState
    :param rounds: Number of rounds.
    :type rounds: int, optional
    :param max_model_size: Maximum model size in MB.
    :type max_model_size: float
    :param n_iters: Number of iterations for inference.
    :type n_iters: int
    :param degree: Maximum clique size.
    :type degree: int
    :param num_marginals: Number of marginals to sample.
    :type num_marginals: int, optional
    :param max_cells: Maximum number of cells in a clique.
    :type max_cells: int
    :param structural_zeros: Structural zeros in the data.
    :type structural_zeros: dict
    :param compress: Whether to compress the data.
    :type compress: bool
    :param domain: The domain of the data.
    :type domain: Domain
    :param n_jobs: Number of parallel jobs.
    :type n_jobs: int
    """

    def __init__(
        self,
        domain: Domain,
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
        """
        Sample a candidate using the provided scores using the exponential mechanism.

        :param qualities: Quality scores for candidates.
        :type qualities: Union[dict, np.ndarray]
        :param epsilon: Privacy budget.
        :type epsilon: float
        :param sensitivity: Sensitivity of the scores. Defaults to 1.0.
        :type sensitivity: float, optional
        :param base_measure: Base measure for candidates. Defaults to None.
        :type base_measure: np.ndarray, optional
        :return: The selected candidate.
        :rtype: Any

        **Example**::

            >>> exponential_mechanism(qualities, 1.0)
            (1, 2)
        """
        if isinstance(qualities, dict):
            keys = list(qualities.keys())
            qualities = np.array([qualities[key] for key in keys])
            if base_measure is not None:
                base_measure = np.log([base_measure[key] for key in keys])
        else:
            qualities = np.array(qualities)
            keys = np.arange(qualities.size)

        q = qualities - qualities.max()
        if base_measure is None:
            p = softmax(0.5 * epsilon / sensitivity * q)
        else:
            p = softmax(0.5 * epsilon / sensitivity * q + base_measure)

        return keys[self.prng.choice(p.size, p=p)]

    def worst_approximated(self, candidates, answers, model, eps, sigma, public=False):
        """
        Find the worst approximated clique using the exponential mechanism.

        :param candidates: Candidate cliques with scores.
        :type candidates: dict
        :param answers: True answers for the cliques.
        :type answers: dict
        :param model: The current graphical model.
        :type model: GraphicalModel
        :param eps: Privacy budget.
        :type eps: float
        :param sigma: Noise scale.
        :type sigma: float
        :param public: Whether the data is public. Defaults to False.
        :type public: bool, optional
        :return: The worst approximated clique.
        :rtype: tuple

        **Example**::

            >>> worst_approximated(candidates, answers, model, 1.0, 0.5)
            (1, 2)
        """
        errors = {}
        sensitivity = {}

        if self.n_jobs == 1:
            for cl in candidates:
                _, errors[cl], sensitivity[cl] = self._worst_approximated(
                    cl, candidates[cl], answers[cl], model, sigma
                )
        else:
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

    def _fit(self, data: Dataset, public=False, workload=None):  # noqa: C901
        """
        Fit the AIM mechanism to the data.

        :param data: The dataset.
        :type data: Dataset
        :param public: Whether the data is public. Defaults to False.
        :type public: bool, optional
        :param workload: Workload cliques. Defaults to None.
        :type workload: list, optional
        :return: The dataset and measurements.
        :rtype: tuple

        **Example**::

            >>> _fit(data, public=True)
            (data, measurements)
        """
        if workload is None:
            workload = list(itertools.combinations(data.domain, self.degree))
            workload = [cl for cl in workload if data.domain.size(cl) <= self.max_cells]
            workload = [(cl, 1.0) for cl in workload]

        if self.num_marginals is not None:
            workload = [
                workload[i]
                for i in self.prng.choice(
                    len(workload), self.num_marginals, replace=False
                )
            ]

        rounds = self.rounds or 16 * len(data.domain)
        workload = [cl for cl, _ in workload]
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

        self._compression_applied = (
            self.compress
            and not self.structural_zeros
        )

        if self._compression_applied:
            measurements = self.compressor.fit(measurements)
            data = self.compressor.transform(data)

        self.engine = FactoredInference(
            data.domain,
            iters=self.n_iters,
            structural_zeros=self.structural_zeros,
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
            q_matrix = Identity(n)
            x = data.project(cl).datavector()
            # TODO: figure out to make determinstic
            if public:
                y = x
            else:
                y = x + gaussian_noise(sigma=sigma, size=n)

            self.cliques.append(cl)
            measurements.append((q_matrix, y, sigma, cl))
            z = model.project(cl).datavector()

            model = self.engine.estimate(measurements)
            w = model.project(cl).datavector()
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                sigma /= 2
                epsilon *= 2

        return data, measurements

    def store(self, path: Path):
        """
        Store the AIM mechanism state to a file.

        :param path: The path to store the state.
        :type path: Path

        **Example**::

            >>> store(Path("/tmp/state"))
        """
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
        """
        Load the AIM mechanism state from a file.

        :param path: The path to load the state.
        :type path: Path
        :return: The loaded AIM mechanism.
        :rtype: AIM

        **Example**::

            >>> AIM.load(Path("/tmp/state"))
        """
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
    """
    Adaptive and Iterative Mechanism (AIM) for differential privacy.
    The model works by iteratively selecting the worst approximated clique
    and adding noise to it.
    It uses the exponential mechanism to select the worst approximated clique
    based on the quality scores of the candidates.

    Ref: https://arxiv.org/pdf/2201.12677

    :param epsilon: Privacy budget.
    :type epsilon: float
    :param delta: Privacy parameter.
    :type delta: float
    :param rounds: Number of rounds.
    :type rounds: int, optional
    :param compress: Whether to compress the data.
    :type compress: bool
    :param n_iters: Number of iterations for inference.
    :type n_iters: int
    :param max_model_size: Maximum model size in MB.
    :type max_model_size: float
    :param degree: Maximum clique size.
    :type degree: int
    :param num_marginals: Number of marginals to sample.
    :type num_marginals: int, optional
    :param max_cells: Maximum number of cells in a clique.
    :type max_cells: int
    :param structural_zeros: Structural zeros in the data.
    :type structural_zeros: dict
    :param domain: The domain of the data.
    :type domain: Domain
    :param random_state: Random state for reproducibility.
    :type random_state: RandomState
    """

    name = "aim"

    def __init__(
        self,
        domain: Domain,
        epsilon: float = 1,
        delta: float = 1e-5,
        rounds: int = None,
        compress: bool = True,
        n_iters: int = 1000,
        max_model_size=80,
        degree=2,
        num_marginals=None,
        max_cells=10000,
        structural_zeros={},
        random_state: RandomState = None,
        n_jobs: int = -1,
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
            n_jobs=n_jobs,
        )

    def set_random_state(self, random_state: RandomState):
        """
        Set the random state for reproducibility.

        :param random_state: The random state.
        :type random_state: RandomState

        **Example**::

            >>> set_random_state(RandomState(42))
        """
        super().set_random_state(random_state)
        self.generator.set_random_state(random_state)

    def set_domain(self, domain: Dict):
        """
        Set the domain of the data.

        :param domain: The domain.
        :type domain: Dict

        **Example**::

            >>> set_domain({"age": [0, 100], "income": [0, 100000]})
        """
        super().set_domain(domain=domain)
        self.generator.set_domain(domain=domain)

    @to_path
    def store(self, path: Path):
        """
        Store the AIMGM model state to a file.

        :param path: The path to store the state.
        :type path: Path

        **Example**::

            >>> store(Path("/tmp/state"))
        """
        super().store(path)
        self.generator.store(path)

    @classmethod
    @to_path
    def load(cls, path: Path) -> Self:
        """
        Load the AIMGM model state from a file.

        :param path: The path to load the state.
        :type path: Path
        :return: The loaded AIMGM model.
        :rtype: AIMGM

        **Example**::

            >>> AIMGM.load(Path("/tmp/state"))
        """
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
