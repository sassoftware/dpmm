# A generative model training algorithm based on
# "PrivBayes: Private Data Release via Bayesian Networks"
# by Jun Zhang1 Graham Cormode2 Cecilia M. Procopiuc3 Divesh Srivastava3 Xiaokui Xiao1
# combined with the pgm framework by Ryan McKenna, Gerome Miklau, Daniel Sheldon
# code inspired from : https://github.com/ryan112358/private-pgm/blob/1da21c8b38149b05f1385b8e54116568b700b4fa/mechanisms/mst.py


from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Self, Tuple

# import argparse
import joblib
import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

from dpmm.models.base.graphical import GraphicalGenerativeModel
from dpmm.models.base.mbi import Dataset, GraphicalModel
from dpmm.models.base.mbi.domain import Domain
from dpmm.models.base.mechanisms import cdp_rho
from dpmm.models.base.memory import clique_size
from dpmm.utils import to_path
from dpmm.models.base.mechanisms import Mechanism
from warnings import warn

"""
This is a generalization of the winning mechanism from the
2018 NIST Differential Privacy Synthetic Data Competition.

Unlike the original implementation, this one can work for any discrete dataset,
and does not rely on public provisional data for measurement selection.
"""


class MI:
    """
    Mutual Information (MI) computation class.

    :param data: The dataset.
    :type data: Dataset
    """

    def __init__(self, data: Dataset):
        self.cache = {}
        self.data = data

    def __call__(self, parents: Tuple[str], child: str) -> float:
        """
        Compute the mutual information between a child and its parents.

        :param parents: Tuple of parent attributes.
        :type parents: Tuple[str]
        :param child: Child attribute.
        :type child: str
        :return: Mutual information value.
        :rtype: float
        """
        parents = tuple(parents)
        cache_key = (child,) + parents

        if cache_key in self.cache:
            return self.cache[cache_key]

        joint_counts = self.data.project(list(parents + (child,))).datavector(
            flatten=False
        )

        joint_counts = np.nan_to_num(joint_counts, nan=0.0)
        normaliser = joint_counts.sum(axis=None)
        if normaliser == 0:
            normaliser = 1

        if child not in self.cache:
            p_1 = joint_counts.sum(axis=tuple(np.arange(len(parents)))) / normaliser
            self.cache[child] = e_1 = self.entropy(p_1)
        else:
            e_1 = self.cache[child]

        if parents not in self.cache:
            p_2 = joint_counts.sum(axis=-1) / normaliser
            self.cache[parents] = e_2 = self.entropy(p_2)
        else:
            e_2 = self.cache[parents]

        p_1_2 = joint_counts / normaliser
        e_1_2 = self.entropy(p_1_2)
        if e_1 == 0 or e_2 == 0:
            value = 0
        else:
            value = (e_1 + e_2 - e_1_2) / np.sqrt(e_1 * e_2)
        self.cache[cache_key] = value
        return value

    def entropy(self, p: np.ndarray) -> float:
        """
        Calculate the entropy of a probability distribution.

        :param p: Probability distribution.
        :type p: np.ndarray
        :return: Entropy value.
        :rtype: float
        """
        pp = p[p > 0]
        return -np.sum(pp * np.log2(pp), axis=None)


class ExponentialMechanism:
    """
    Exponential Mechanism for differential privacy.

    :param data: The dataset.
    :type data: Dataset
    :param epsilon: Privacy budget.
    :type epsilon: float
    :param normalise: Whether to normalise the output probabilities.
    :type normalise: bool
    """

    def __init__(self, data: Dataset, epsilon: float, normalise: bool = True):
        self.epsilon = epsilon
        self.delta = self.compute_delta(data)
        self.normalise = normalise

    def compute_delta(self, data: Dataset) -> float:
        """
        Compute the sensitivity delta for the dataset.

        :param data: The dataset.
        :type data: Dataset
        :return: Sensitivity delta.
        :rtype: float
        """
        n_rows = data.df.shape[0]
        n_nodes = len(data.domain.attrs)
        a = (2 / n_rows) * np.log((n_rows + 1) / 2)
        b = (1 - 1 / n_rows) * np.log(1 + 2 / (n_rows - 1))
        sensitivity = a + b
        eps = self.epsilon / n_nodes
        delta = sensitivity / eps
        return delta

    def __call__(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Apply the exponential mechanism to an input vector.

        :param input_vector: Input vector.
        :type input_vector: np.ndarray
        :return: Output probabilities.
        :rtype: np.ndarray
        """
        with_delta = np.array(input_vector) / (2 * self.delta)

        if self.normalise:
            log_normalised = with_delta - logsumexp(with_delta)
            normalised = np.exp(log_normalised)
            normalised /= normalised.sum()
            return normalised

        return np.exp(with_delta)


class PrivBayes(Mechanism):
    """
    PrivBayes is a differentially private generative model that uses Bayesian networks
    to approximate the joint distribution of the data. It selects dependencies between
    attributes based on mutual information and relies on
    the exponential and laplace mechanism to ensure differential privacy.

    Ref: https://dl.acm.org/doi/pdf/10.1145/3134428

    :param epsilon: Privacy budget.
    :type epsilon: float
    :param delta: Privacy parameter.
    :type delta: float, optional
    :param degree: Maximum degree of the Bayesian network. The size of the largest clique.
    :type degree: int
    :param n_iters: Number of iterations for fitting.
    :type n_iters: int
    :param max_candidates: Maximum number of candidates for selection.
    :type max_candidates: int
    :param n_jobs: Number of parallel jobs.
    :type n_jobs: int
    :param prng: Random state for reproducibility.
    :type prng: RandomState, optional
    :param max_model_size: Maximum size of the model.
    :type max_model_size: int, optional
    :param compress: Whether to compress the data.
    :type compress: bool
    :param domain: The domain of the data.
    :type domain: Domain, optional
    :param structural_zeros: Structural zeros in the data.
    :type structural_zeros: Dict, optional
    """

    def __init__(
        self,
        domain: Domain,
        epsilon=1,
        delta=None,
        degree=2,
        n_iters: int = 5000,
        max_candidates: int = 20,
        n_jobs: int = -1,
        prng: RandomState = None,
        max_model_size: int = None,
        compress=False,
        structural_zeros: Dict = None,
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

        self.rho = cdp_rho(self.epsilon / 2, self.delta)
        self.sigma = np.sqrt(1 / (2 * self.rho))

        self.degree = degree
        self.n_iters = n_iters
        self.max_candidates = max_candidates

    def select(self, data: Dataset, public=False) -> list:  # noqa: C901
        """
        Select cliques based on mutual information.

        :param data: The dataset.
        :type data: Dataset
        :param public: Whether to use public data for selection.
        :type public: bool
        :return: List of selected cliques.
        :rtype: list
        """
        remaining = set(data.domain.attrs)
        used = set()
        cliques = []

        attrs = data.domain.attrs
        if self.max_model_size is not None:
            size_per_node = self.max_model_size / len(attrs)
            attrs = [
                attr
                for attr in attrs
                if clique_size(data, clique=(attr)) <= size_per_node
            ]
            if len(attrs) == 0:
                attrs = data.domain.attrs
        root = self.prng.choice(attrs)
        used.add(root)
        remaining.remove(root)
        cliques.append((root,))

        mi = MI(data=data)
        if not public:
            exp_mechanism = ExponentialMechanism(
                data=data, epsilon=self.epsilon / 2, normalise=True
            )

        while len(remaining) > 0:
            degree = min(len(used), self.degree - 1)
            candidates = []
            combs = list(combinations(used, degree))
            for col in remaining:
                col_candidates = combs
                if self.max_model_size is not None:
                    col_candidates = [
                        candidate
                        for candidate in col_candidates
                        if clique_size(data, clique=((candidate) + (col,)))
                        <= size_per_node
                    ]
                if len(col_candidates) == 0:
                    min_size = min(
                        [
                            clique_size(data, clique=((candidate) + (col,)))
                            for candidate in col_candidates
                        ]
                    )
                    warn(
                        f"No combination respects the clique size limitation, will be skipped. Min Size : {min_size} | expected : {size_per_node}"
                    )
                    col_candidates = list(combs)

                if self.max_candidates is not None:
                    if self.max_candidates < len(col_candidates):
                        selected_idx = list(
                            self.prng.choice(
                                len(col_candidates),
                                size=self.max_candidates,
                                replace=False,
                            )
                        )
                        col_candidates = [col_candidates[idx] for idx in selected_idx]

                candidates += [(candidate, col) for candidate in col_candidates]

            if self.n_jobs > 1:
                mi_scores = np.array(
                    [mi_score for mi_score in Pool(self.n_jobs).starmap(mi, candidates)]
                )
            else:
                mi_scores = np.array([mi(*candidate) for candidate in candidates])

            if public:
                selected = np.argmax(mi_scores)
            else:
                p = exp_mechanism(mi_scores)
                selected = self.prng.choice(len(candidates), p=p, size=1)[0]

            selected_candidate, selected_col = candidates[selected]
            selected_clique = selected_candidate + (selected_col,)

            cliques.append(selected_clique)
            used.add(selected_col)
            remaining.remove(selected_col)

        return cliques

    def _fit(self, data: Dataset, public=False) -> Tuple[Dataset, np.ndarray]:
        """
        Fit the PrivBayes model to the data.

        :param data: The dataset.
        :type data: Dataset
        :param public: Whether to use public data for fitting.
        :type public: bool
        :return: Transformed data and measures.
        :rtype: Tuple[Dataset, np.ndarray]
        """
        self.cliques = self.select(data=data, public=public)
        measures = self.measure(data, public=public, flatten=not self.compress)

        if self.compress:
            measures = self.compressor.fit(measures, flatten=True)
            data = self.compressor.transform(data)

        return data, measures

    def store(self, path: Path) -> None:
        """
        Store the PrivBayes model state to a file.

        :param path: The path to store the state.
        :type path: Path
        """
        joblib.dump(
            {
                "epsilon": self.epsilon,
                "delta": self.delta,
                "n_iters": self.n_iters,
                "cliques": self.cliques,
                "compress": self._compress,
                "compressor": self.compressor,
                "degree": self.degree,
                "max_candidates": self.max_candidates,
                "domain": self._domain,
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
        """
        Load the PrivBayes model state from a file.

        :param path: The path to load the state.
        :type path: Path
        :return: The loaded PrivBayes model.
        :rtype: PrivBayes
        """
        state = joblib.load(path / "state.joblib")
        obj = cls(
            epsilon=state["epsilon"],
            delta=state["delta"],
            n_iters=state["n_iters"],
            degree=state["degree"],
            max_candidates=state["max_candidates"],
            compress=state["compress"],
            domain=state["domain"],
            max_model_size=state["max_model_size"],
        )

        obj.cliques = state["cliques"]
        obj.compressor = state["compressor"]
        obj.model_size = state["model_size"]
        obj.fit_state = state["fit_state"]

        model_path = path / "estimator.pickle"
        if model_path.exists():
            obj.model = GraphicalModel.load(model_path)
        return obj


class PrivBayesGM(GraphicalGenerativeModel):
    """
    PrivBayesGM is a wrapper for the PrivBayes generative model.

    :param epsilon: Privacy budget.
    :type epsilon: float
    :param delta: Privacy parameter.
    :type delta: float
    :param degree: Maximum degree of the Bayesian network.
    :type degree: int
    :param n_iters: Number of iterations for fitting.
    :type n_iters: int
    :param compress: Whether to compress the data.
    :type compress: bool
    :param max_model_size: Maximum size of the model.
    :type max_model_size: int, optional
    :param domain: The domain of the data.
    :type domain: Domain, optional
    :param random_state: Random state for reproducibility.
    :type random_state: RandomState, optional
    """

    name = "priv-bayes"
    mechansim = PrivBayes

    def __init__(
        self,
        domain: Domain,
        epsilon: float = 1,
        delta: float = 1e-5,
        degree: int = 2,
        n_iters: int = 5000,
        compress: bool = True,
        max_model_size: int = None,
        random_state: RandomState = None,
        n_jobs: int = -1,
    ):
        super().__init__(domain=domain, random_state=random_state)

        self.epsilon = epsilon
        self.delta = delta

        self.generator = self.mechansim(
            epsilon=self.epsilon,
            delta=self.delta,
            n_iters=n_iters,
            degree=degree,
            compress=compress,
            domain=domain,
            prng=random_state,
            max_model_size=max_model_size,
            n_jobs=n_jobs,
        )

    def set_random_state(self, random_state: RandomState) -> None:
        """
        Set the random state for reproducibility.

        :param random_state: Random state.
        :type random_state: RandomState
        """
        super().set_random_state(random_state)
        self.generator.set_random_state(random_state)

    def set_domain(self, domain: Dict) -> None:
        """
        Set the domain of the data.

        :param domain: The domain.
        :type domain: Dict
        """
        super().set_domain(domain=domain)
        self.generator.set_domain(domain=domain)

    @to_path
    def store(self, path: Path) -> None:
        """
        Store the PrivBayesGM model state to a file.

        :param path: The path to store the state.
        :type path: Path
        """
        super().store(path)
        self.generator.store(path)

    @classmethod
    @to_path
    def load(cls, path: Path) -> Self:
        """
        Load the PrivBayesGM model state from a file.

        :param path: The path to load the state.
        :type path: Path
        :return: The loaded PrivBayesGM model.
        :rtype: PrivBayesGM
        """
        generator = PrivBayes.load(path)
        obj = cls(
            epsilon=generator.epsilon, delta=generator.delta, domain=generator._domain
        )
        del obj.generator
        obj.generator = generator

        return obj
