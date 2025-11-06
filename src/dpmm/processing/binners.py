import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Union
from numpy.random import RandomState

from dpmm.models.base.utils import laplace_noise
from dpmm.processing.utils import approx_bounds, optimal_n_bins

cast_map = {
    "f": np.float32,
    "i": np.int32,
    "u": np.uint32,
    "b": np.bool_,
    "M": np.datetime64,
    "m": lambda x: pd.Timedelta(x).to_numpy(),
}


def get_cast(dtype: np.dtype) -> Union[np.dtype, callable]:
    """
    Get the appropriate casting function or dtype for a given numpy dtype.

    :param dtype: The numpy dtype to map.
    :return: The corresponding casting function or dtype.

    .. code-block:: python

        dtype = np.dtype('float32')
        cast_func = get_cast(dtype)
    """
    return cast_map[dtype.kind]


class Binner:
    name: str = None

    def __init__(
        self,
        n_bins: Union[str, int] = "auto",
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        epsilon: Optional[float] = None,
        max_n_bins: Optional[int] = None,
        rnd: Optional[RandomState] = None,
        structural_zeros: Optional[List[Tuple[float, float]]] = None,
    ):
        """
        Initialize the Binner.

        :param n_bins: Number of bins or 'auto' for automatic binning.
        :param lower: Lower bound of the data.
        :param upper: Upper bound of the data.
        :param epsilon: Privacy budget.
        :param max_n_bins: Maximum number of bins.
        :param rnd: Random state for reproducibility.
        :param structural_zeros: List of intervals treated as structural zeros.

        .. code-block:: python

            binner = Binner(n_bins=10, lower=0, upper=1, epsilon=0.1)
        """
        self.n_bins = n_bins
        self.lower = lower
        self.upper = upper
        self.epsilon = epsilon
        self.spent_epsilon = None
        self.max_n_bins = max_n_bins
        self.dtype = None
        if structural_zeros is None:
            structural_zeros = []

        self.set_structural_zeros(structural_zeros)

        # Random State
        if rnd is None:
            rnd = RandomState()
        self.rnd = rnd

    def set_structural_zeros(self, structural_zeros: List[Tuple[float, float]]) -> None:
        """
        Set structural zeros for the binner.

        :param structural_zeros: A list of tuples representing intervals to be treated as structural zeros.

        .. code-block:: python

            binner.set_structural_zeros([(0.1, 0.2), (0.5, 0.6)])
        """
        structural_zeros = sorted(structural_zeros)
        # remove overlapping zeros
        if len(structural_zeros) >= 1:
            new_zeros = [structural_zeros[0]]
            for zero in structural_zeros[1:]:
                # if previous interval is not overlapping with the current interval
                if zero[0] > new_zeros[-1][1]:
                    new_zeros.append(zero)
                elif zero[1] > new_zeros[-1][1]:
                    new_zeros[-1] = (new_zeros[-1][0], zero[1])

            structural_zeros = new_zeros

        self.structural_zeros = structural_zeros

    def set_random_state(self, rnd: RandomState) -> None:
        """
        Set the random state for reproducibility.

        :param rnd: A numpy RandomState instance.

        .. code-block:: python

            rnd = RandomState(42)
            binner.set_random_state(rnd)
        """
        self.rnd = rnd

    def spend_epsilon(self, eps: float) -> float:
        """
        Spend a portion of the privacy budget (epsilon).

        :param eps: The amount of privacy budget (epsilon) to spend.
        :return: The spent privacy budget.

        .. code-block:: python

            spent_eps = binner.spend_epsilon(0.1)
        """
        if self.epsilon is None:
            raise ValueError("Should not spend any budget as the overall epsilon")
        else:
            if self.spent_epsilon is None:
                self.spent_epsilon = eps
            else:
                self.spent_epsilon += eps
        return eps

    @property
    def zeros(self) -> np.ndarray:
        """
        Get the indices of bins that are structural zeros.

        :return: An array of indices of structural zero bins.

        .. code-block:: python

            zero_indices = binner.zeros
        """
        col_zeros = np.array(self.structural_zeros)  # n_zeroes x 2 (start, end)
        bins = np.array([self.bins[:-1], self.bins[1:]]).T  # n_bins x 2 (start, end)
        bin_is_in_interval = np.logical_and(
            col_zeros[np.newaxis, :, 0] <= bins[:, np.newaxis, 0],
            col_zeros[np.newaxis, :, 1] >= bins[:, np.newaxis, 1],
        )  # n_bins x n_zeroes
        is_zero = np.any(bin_is_in_interval, axis=1)
        return np.arange(len(is_zero))[is_zero]

    def get_bounds(
        self, arr: np.ndarray, recompute: bool = False, epsilon: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute or retrieve the bounds of the data.

        :param arr: The input array.
        :param recompute: Whether to recompute the bounds.
        :param epsilon: The privacy budget for computing bounds.
        :return: A tuple containing the lower and upper bounds.

        .. code-block:: python

            bounds = binner.get_bounds(arr, epsilon=0.1)
        """

        if self.has_bounds and not recompute:
            min_, max_ = self.lower, self.upper
        elif epsilon is None:
            min_, max_ = arr.min(), arr.max()
            if pd.notnull(self.lower):
                min_ = self.lower
            if pd.notnull(self.upper):
                max_ = self.upper
        else:
            min_, max_ = approx_bounds(arr, epsilon=self.spend_epsilon(epsilon))

        type_cast = get_cast(self.dtype)
        min_, max_ = type_cast(min_), type_cast(max_)
        return min_, max_

    def bin_bounds(self, bin_idx: int) -> Tuple[float, float]:
        """
        Get the bounds of a specific bin.

        :param bin_idx: The index of the bin.
        :return: A tuple containing the lower and upper bounds of the bin.

        .. code-block:: python

            bin_bounds = binner.bin_bounds(3)
        """
        return self.bins[bin_idx], self.bins[bin_idx + 1]

    @property
    def has_bounds(self) -> bool:
        """
        Check if the binner has predefined bounds.

        :return: True if bounds are defined, False otherwise.

        .. code-block:: python

            has_bounds = binner.has_bounds
        """
        return (self.lower is not None) and (self.upper is not None)

    def fit(self, arr: np.ndarray) -> None:
        """
        Fit the binner to the data.

        :param arr: The input array to fit.

        .. code-block:: python

            binner.fit(arr)
        """
        if isinstance(self.n_bins, str):
            self.n_bins = optimal_n_bins(
                arr, upper=self.max_n_bins, epsilon=self.epsilon
            )
        self.dtype = arr.dtype

    def transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Transform the data into bins.

        :param arr: The input array to transform.
        :return: The transformed array.

        .. code-block:: python

            transformed = binner.transform(arr)
        """
        raise NotImplementedError

    def fit_transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Fit the binner to the data and transform it.

        :param arr: The input array to fit and transform.
        :return: The transformed array.

        .. code-block:: python

            transformed = binner.fit_transform(arr)
        """
        self.fit(arr)
        return self.transform(arr)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Inverse transform the binned data back to its original form.

        :param arr: The binned array to inverse transform.
        :return: The original array.

        .. code-block:: python

            original = binner.inverse_transform(binned_arr)
        """
        raise NotImplementedError

    @property
    def bin_domain(self) -> int:
        """
        Get the number of bins minus one.

        :return: The number of bins minus one.

        .. code-block:: python

            domain = binner.bin_domain
        """
        return len(self.bins) - 1


class UniformBinner(Binner):
    name: str = "uniform"

    def __init__(
        self,
        n_bins: int = 20,
        epsilon: Optional[float] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        max_n_bins: Optional[int] = None,
        rnd: Optional[RandomState] = None,
    ):
        """
        Initialize the UniformBinner. Distribute the data into uniform bins.
        The number of bins is determined by the `n_bins` parameter.

        :param n_bins: Number of bins.
        :param epsilon: Privacy budget.
        :param lower: Lower bound of the data.
        :param upper: Upper bound of the data.
        :param max_n_bins: Maximum number of bins.
        :param rnd: Random state for reproducibility.

        .. code-block:: python

            binner = UniformBinner(n_bins=10)
        """
        super().__init__(
            n_bins=n_bins,
            lower=lower,
            upper=upper,
            epsilon=epsilon,
            max_n_bins=max_n_bins,
            rnd=rnd,
        )
        self.bins = None
        self.stats = {}

    def fit(self, arr: np.ndarray) -> None:
        """
        Fit the UniformBinner to the data.

        :param arr: The input array to fit.

        .. code-block:: python

            binner.fit(arr)
        """
        super().fit(arr)

        if self.epsilon is None:
            eps = None
        else:
            eps = self.epsilon

        lower_, upper_ = self.get_bounds(arr=arr, epsilon=eps)
        self.bins = lower_ + (upper_ - lower_) * np.linspace(0, 1, num=self.n_bins + 1)

    def transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Transform the data into bins.

        :param arr: The input array to transform.
        :return: The transformed array.

        .. code-block:: python

            transformed = binner.transform(arr)
        """

        bins = np.array(self.bins).astype(arr.dtype)
        labels = [
            idx for idx, bin in enumerate(self.bins[:-1]) if self.bins[idx + 1] != bin
        ]

        t_series = (
            pd.Series(
                pd.cut(
                    arr.clip(bins[0], bins[-1]),
                    bins=self.bins,
                    right=False,
                    labels=labels,
                    duplicates="drop",
                )
            )
            .fillna(labels[-1])
            .astype(int)
            .clip(lower=0, upper=(len(self.bins) - 2))
        )

        return t_series.to_numpy()

    def get_intervals(self, bin_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the intervals and weights for a specific bin.

        :param bin_idx: The index of the bin.
        :return: A tuple containing the intervals and their weights.

        .. code-block:: python

            intervals, weights = binner.get_intervals(3)
        """
        lower, upper = self.bins[bin_idx], self.bins[bin_idx + 1]
        intervals = np.array([(lower, upper)])

        if len(self.structural_zeros) > 0:
            left_zeros = np.array([zero[0] for zero in self.structural_zeros])
            right_zeros = np.array([zero[1] for zero in self.structural_zeros])

            # if lower is in an interval that stops before upper bound
            lower_in_zero = (
                (lower >= left_zeros) & (lower <= right_zeros) & (right_zeros <= upper)
            )
            if lower_in_zero.any():
                lower = right_zeros[lower_in_zero].max()

            # update upper
            upper_in_zero = (
                (lower <= left_zeros) & (upper >= left_zeros) & (upper <= right_zeros)
            )
            if upper_in_zero.any():
                upper = left_zeros[upper_in_zero].min()

            # Create new intervals to sample from
            _steps = (
                [lower]
                + [
                    z
                    for zero in self.structural_zeros
                    for z in zero
                    if (zero[0] >= lower) and (zero[1] <= upper)
                ]
                + [upper]
            )
            intervals = np.array(
                [
                    (step, _steps[idx + 1])
                    for idx, step in list(enumerate(_steps[:-1]))[::2]
                ]
            )

        weights = np.array([z[1] - z[0] for z in intervals])
        weights = weights / weights.sum()
        return intervals, weights

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Inverse transform the binned data back to its original form.

        :param arr: The binned array to inverse transform.
        :return: The original array.

        .. code-block:: python

            original = binner.inverse_transform(binned_arr)
        """
        i_series = pd.Series(index=np.arange(arr.shape[0]), dtype=object)
        for bin_idx in range(self.n_bins):
            intervals, weights = self.get_intervals(bin_idx)

            bin_match = arr == bin_idx
            int_idx = self.rnd.choice(len(intervals), size=bin_match.sum(), p=weights)
            new_series = (
                self.rnd.uniform(0, 1, size=bin_match.sum())
                * (intervals[int_idx, 1] - intervals[int_idx, 0])
                + intervals[int_idx, 0]
            )
            # if "timedelta" in self.dtype.name:
            i_series.loc[bin_match] = new_series

        if self.dtype.kind in "ui":
            i_series = i_series.astype(float).round().astype(self.dtype)
        return i_series.to_numpy()


class PrivTreeBinner(UniformBinner):
    name: str = "priv-tree"
    beta = 2

    def __init__(
        self,
        n_bins: int = 20,
        enforce_bins: bool = True,
        epsilon: Optional[float] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        max_n_bins: Optional[int] = None,
        decay: float = 0,
        rnd: Optional[RandomState] = None,
    ):
        """
        Initialize the PrivTreeBinner.
        Based on the (PrivTree Algorithm)[https://arxiv.org/pdf/1601.03229].


        :param n_bins: Number of bins.
        :param enforce_bins: Whether to enforce the number of bins.
        :param epsilon: Privacy budget.
        :param lower: Lower bound of the data.
        :param upper: Upper bound of the data.
        :param max_n_bins: Maximum number of bins.
        :param decay: Decay factor for the privacy budget.
        :param rnd: Random state for reproducibility.

        .. code-block:: python

            binner = PrivTreeBinner(n_bins=10, enforce_bins=True, epsilon=0.1)
        """
        super().__init__(
            n_bins=n_bins,
            lower=lower,
            epsilon=epsilon,
            upper=upper,
            max_n_bins=max_n_bins,
            rnd=rnd,
        )
        self.bins = None
        self.enforce_bins = enforce_bins
        self.theta = None
        self.decay = decay
        self.scale = None

    def priv_tree(
        self, data: np.ndarray, eps: Optional[float] = None
    ) -> Tuple[Dict, Dict, Dict, Dict, List[str]]:
        """
        Builds bin by splitting the data into a binary tree structure.
        The tree is built by recursively splitting the data into two halves
        until the number of data points in a node is less than theta.
        The tree is built in a top-down manner, where each node represents
        a sub-domain of the data.
        The tree is built using the Laplace mechanism to ensure differential privacy.
        The tree is built in a way that the number of bins is equal to n_bins.

        :param data: The input data array.
        :param eps: The privacy budget for the tree.
        :return: A tuple containing domains, children, counts, depths, and leaf nodes.

        .. code-block:: python

            domains, children, counts, depths, leaf_nodes = binner.priv_tree(data, eps=0.1)
        """
        if len(data.shape) == 1:
            lower_, upper_ = self.get_bounds(arr=data, epsilon=eps)
            data = data[:, np.newaxis]
            lower_, upper_ = np.array([lower_]), np.array([upper_])
        else:
            # TODO: Add get_bounds in multi dimensions
            lower_, upper_ = data.min(axis=0), data.max(axis=0)

        domains = {"v_0": (lower_, upper_, data)}
        children = {}
        counts = {"v_0": data.shape[0]}
        depths = {"v_0": 0}
        node_idx = 1
        unvisited = ["v_0"]
        leaf_nodes = []

        while unvisited:
            # get unvisited node
            node = unvisited[0]  # node name
            node_count = counts[node]  # node sub-domain count
            node_depth = depths[node]  # node depth
            node_domain = domains[node]

            # remove node from unvisited
            unvisited = unvisited[1:]

            # biased count
            b_node = max(
                (node_count - (node_depth * self.decay)), (self.theta - self.decay)
            )

            # add laplace noise
            if self.scale is not None:
                b_node += laplace_noise(scale=self.scale)

            if (b_node > self.theta) and (
                (not self.enforce_bins)
                or ((len(leaf_nodes) + len(unvisited) + 2) <= self.n_bins)
            ):
                node_split = node_domain[0] + (node_domain[1] - node_domain[0]) / 2
                children[node] = []

                # Split domain
                child_domains = [node_domain]
                for axis_idx in range(node_split.shape[0]):
                    new_domains = []
                    for domain_lower, domain_upper, domain_data in child_domains:
                        node_split = domain_lower + (domain_upper - domain_lower) / 2
                        split_flag = domain_data[:, axis_idx] < node_split[axis_idx]
                        left = domain_data[split_flag]
                        right = domain_data[~split_flag]
                        new_domains += [
                            (domain_lower, node_split, left),
                            (node_split, domain_upper, right),
                        ]

                    child_domains = new_domains

                # Update list
                for child_node in child_domains:
                    node_name = f"v_{node_idx}"
                    # Update children
                    children[node].append(node_name)
                    # Update domains
                    domains[node_name] = child_node
                    # Update counts
                    counts[node_name] = child_node[-1].shape[0]
                    # Update depth
                    depths[node_name] = node_depth + 1
                    # Update unvisited
                    unvisited.append(node_name)
                    # Update node index
                    node_idx += 1
            else:
                leaf_nodes.append(node)

        return domains, children, counts, depths, leaf_nodes

    def fit(self, arr: np.ndarray) -> None:
        """
        Fit the PrivTreeBinner to the data.

        :param arr: The input array to fit.

        .. code-block:: python

            binner.fit(arr)
        """
        Binner.fit(self, arr)
        self.theta = arr.shape[0] / self.n_bins
        eps = self.epsilon
        if self.epsilon:
            eps = self.epsilon / (1 + int(not (self.has_bounds)))

            self.scale = (2 * self.beta - 1) / (
                (self.beta - 1) * self.spend_epsilon(eps)
            )  # ln(2) to be changed to ln(2^dim)
            self.decay = self.scale * np.log(2)  # ln(2) to be changed to ln(2^dim)

        (
            self.domains,
            self.children,
            self.counts,
            self.depths,
            self.leaf_nodes,
        ) = self.priv_tree(data=arr, eps=eps)

        sorted_leaves = sorted(self.leaf_nodes, key=lambda x: tuple(self.domains[x][0]))
        self.bins = [self.domains[node][0][0] for node in sorted_leaves] + [
            self.domains[sorted_leaves[-1]][1][0]
        ]

        self.n_bins = len(self.bins) - 1
