from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Self, Tuple

import joblib
import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.preprocessing import OrdinalEncoder

from dpmm.processing import BINNER_DICT
from dpmm.utils import to_path

logger = getLogger("dpmm")


class TableBinner:
    def __init__(
        self,
        binner_type: str = "priv-tree",
        binner_settings: Dict = {"n_bins": "auto"},
        domain: Optional[Dict] = None,
        structural_zeros: Optional[Dict[str, Tuple]] = None,
        random_state: Optional[RandomState] = None,
    ):
        """
        Initialize the TableBinner. This class is used to bin numerical and categorical columns in a DataFrame.
        It supports two binning strategies and can handle missing values and structural zeros.
        Categorical columns are encoded using OrdinalEncoder.
        Static columns are dropped from the DataFrame.

        :param binner_type: The type of binner to use (e.g., "priv-tree").
        :param binner_settings: Settings for the binner.
        :param domain: Optional domain information for the data.
        :param structural_zeros: Optional structural zeros for specific columns.
        :param random_state: Optional random state for reproducibility.

        .. code-block:: python

            binner = TableBinner(binner_type="uniform", binner_settings={"n_bins": 10})
        """
        self.binner_type = binner_type
        self.binner_settings = binner_settings

        if domain is None:
            domain = {}
        self.domain = domain
        if random_state is None:
            random_state = RandomState()
        self.random_state = random_state

        self.binners = None
        self.dtypes = None
        self.oh_encoder = None
        self.bin_domain = None
        self.nan_columns = None
        self.static_columns = None
        self.col_order = None
        self.categories = None
        self.is_fit = False

        if structural_zeros is None:
            structural_zeros = {}
        self.set_structural_zeros(structural_zeros)

    def set_params(self, **kwargs) -> None:
        """
        Update binner settings with provided parameters.

        :param kwargs: Key-value pairs of parameters to update.

        .. code-block:: python

            binner.set_params(n_bins=15, epsilon=0.1)
        """
        self.binner_settings.update(kwargs)

    def set_domain(self, domain: Dict) -> None:
        """
        Set the domain for the binner.

        :param domain: A dictionary specifying the domain for each column.

        .. code-block:: python

            binner.set_domain({"col1": {"lower": 0, "upper": 10}})
        """
        if domain is not None:
            self.domain = domain

    def set_structural_zeros(self, structural_zeros: Dict[str, Tuple]) -> None:
        """
        Set the structural zeros for the binner.

        :param structural_zeros: A dictionary mapping columns to structural zero intervals.

        .. code-block:: python

            binner.set_structural_zeros({"col1": [(0.1, 0.2), (0.5, 0.6)]})
        """
        self.structural_zeros = structural_zeros
        if self.binners is not None:
            for col, binner in self.binners.items():
                if col in structural_zeros:
                    binner.set_structural_zeros(structural_zeros[col])

    def set_random_state(self, rnd: RandomState) -> None:
        """
        Set the random state for reproducibility.

        :param rnd: A numpy RandomState instance.

        .. code-block:: python

            rnd = RandomState(42)
            binner.set_random_state(rnd)
        """
        self.random_state = rnd

        if self.binners is not None:
            for _, binner in self.binners.items():
                binner.set_random_state(rnd)

    def get_categories(
        self, col: str, series: pd.Series, public: bool = False
    ) -> List[str]:
        """
        Get categories for a column, handling missing domain information.

        :param col: The column name.
        :param series: The pandas Series for the column.
        :param public: Whether the data is public (no privacy concerns).
        :return: A list of categories.

        .. code-block:: python

            categories = binner.get_categories("col1", df["col1"])
        """
        col_domain = self.domain.get(col, {})
        if "categories" in col_domain:
            categories = col_domain["categories"]
        else:
            if not (public):
                logger.warning(
                    f"PrivacyLeakage: No categorical domain provided for Column {col} - will be imputed."
                )
            categories = series.unique().tolist()

        categories = [cat for cat in categories if pd.notnull(cat)]

        all_numerical = all(
            [
                (
                    np.issubdtype(np.dtype(type(cat)), np.floating)
                    or np.issubdtype(np.dtype(type(cat)), np.integer)
                    or np.issubdtype(np.dtype(type(cat)), np.bool_)
                )
                for cat in categories
            ]
        )

        if all_numerical:
            categories = sorted(categories)
        else:
            categories = sorted(categories, key=lambda x: str(x))

        return categories

    def insert_col(
        self, df: pd.DataFrame, col: str, series: pd.Series
    ) -> Tuple[pd.DataFrame, str]:
        """
        Insert a new column into the DataFrame, ensuring no name conflicts.

        :param df: The input DataFrame.
        :param col: The name of the column to insert.
        :param series: The pandas Series to insert as a column.
        :return: A tuple containing the updated DataFrame and the final column name.

        .. code-block:: python

            df, col_name = binner.insert_col(df, "new_col", pd.Series([1, 2, 3]))
        """
        idx = 0
        while col in df.columns:
            col = f"{col}_{idx}"
            idx += 1

        return pd.concat([df, series.rename(col)], axis=1), col

    def fit(self, df: pd.DataFrame, public: bool = False) -> None:
        """
        Fit the binner to the DataFrame.

        :param df: The input DataFrame.
        :param public: Whether the data is public (no privacy concerns).

        .. code-block:: python

            binner.fit(df, public=True)
        """
        self.dtypes = df.dtypes
        self.binners = {}
        self.cat_encoders = {}
        self.bin_domain = {}
        self.nan_columns = {}
        self.static_columns = {}
        self.categories = {}
        self.has_nan = {}
        self.col_order = df.columns.tolist()

        # NaN Management
        for col, series in df.items():
            na_flag = series.isna()
            not_na_flag = ~na_flag
            if na_flag.any() and not_na_flag.any():
                fill_value = (
                    df.loc[not_na_flag, col]
                    .sample(n=1, random_state=self.random_state)
                    .iloc[0]
                )
                df, nan_col = self.insert_col(
                    df=df, col=f"{col}_NaN", series=na_flag.astype("category")
                )

                self.domain[nan_col] = {"categories": [False, True]}
                df.loc[na_flag, col] = fill_value
                self.nan_columns[col] = {"name": nan_col, "fill_value": fill_value}

        # Static Columns
        self.static_columns = {
            col: series.iloc[0]
            for col, series in df.items()
            if (series.nunique(dropna=False) == 1)
        }
        # Drop Static Columns
        df = df.drop(columns=[col for col in self.static_columns if col in df.columns])

        # Numerical Columns
        self.num_cols = [
            col
            for col, series in df.items()
            if ((series.dtype.kind) in "Mmfui" and (series.dropna().nunique() > 1))
        ]

        # Compute Epsilon
        if public:
            epsilon = None
        else:
            epsilon = self.binner_settings.get("epsilon", None)
            if epsilon is not None and len(self.num_cols) > 0:
                # Split the epsilon
                epsilon /= len(self.num_cols)

        # Categorical Columns
        self.cat_cols = [col for col in df.columns if col not in self.num_cols]
        for col in self.cat_cols:
            categories = self.get_categories(col, df[col], public=public)
            self.bin_domain[col] = len(categories)
            self.categories[col] = categories
            self.cat_encoders[col] = OrdinalEncoder(categories=[categories])
            self.cat_encoders[col] = OrdinalEncoder(categories=[categories])
            self.cat_encoders[col].fit(df[col].to_frame())

        for col in self.num_cols:
            bin_settings = dict(self.binner_settings)
            bin_settings["epsilon"] = epsilon
            if self.domain is not None:
                bin_settings.update(self.domain.get(col, {}))

            self.binners[col] = BINNER_DICT[self.binner_type](
                **bin_settings, rnd=self.random_state
            )
            # Set Structural Zeros
            if self.structural_zeros is not None:
                if col in self.structural_zeros:
                    self.binners[col].set_structural_zeros(self.structural_zeros[col])

            self.binners[col].fit(df[col].to_numpy())
            self.bin_domain[col] = self.binners[col].bin_domain

        self.is_fit = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame using the fitted binners and encoders.

        :param df: The input DataFrame.
        :return: The transformed DataFrame.

        .. code-block:: python

            transformed_df = binner.transform(df)
        """
        # Add NaN columns
        df = df.copy()
        for col, nan_col in self.nan_columns.items():
            na_flag = df[col].isna().rename(nan_col["name"])
            df.loc[na_flag, col] = nan_col["fill_value"]
            df = pd.concat([df, na_flag], axis=1)

        # Drop Static Columns
        df = df.drop(columns=[col for col in self.static_columns if col in df.columns])

        # Transform the DataFrame using the fitted binners and encoders
        dfs = [
            pd.Series(
                self.cat_encoders[col].transform(df[[col]]).squeeze(),
                index=df.index,
                name=col,
                dtype=int,
            )
            for col in self.cat_cols
            if col in df.columns
        ]
        dfs += [
            pd.Series(
                self.binners[col].transform(df[col].to_numpy()),
                index=df.index,
                name=col,
                dtype=int,
            )
            for col in self.num_cols
            if col in df.columns
        ]

        t_df = pd.concat(dfs, axis=1)
        return t_df

    @property
    def n_bins(self) -> Dict[str, int]:
        """
        Get the number of bins for each column.

        :return: A dictionary mapping column names to the number of bins.

        .. code-block:: python

            bins = binner.n_bins
        """
        return {
            col: binner.n_bins
            for col, binner in self.binners.items()
            if not isinstance(binner, dict)
        }

    @property
    def spent_epsilon(self) -> Dict[str, Optional[float]]:
        """
        Get the spent privacy budget (epsilon) for each column.

        :return: A dictionary mapping column names to the spent privacy budget.

        .. code-block:: python

            spent_eps = binner.spent_epsilon
        """
        return {
            col: binner.spent_epsilon
            for col, binner in self.binners.items()
            if not isinstance(binner, dict)
        }

    @property
    def zeros(self) -> Dict[str, List[int]]:
        """
        Get the indices of structural zero bins for each column.

        :return: A dictionary mapping column names to lists of structural zero indices.

        .. code-block:: python

            zero_indices = binner.zeros
        """
        _zeros = {}
        for col, col_zeros in self.structural_zeros.items():
            if col in self.cat_encoders:
                _zeros[col] = [
                    self.cat_encoders[col].transform([[zero]])[0, 0]
                    for zero in col_zeros
                ]
            elif col in self.binners:
                self.binners[col].set_structural_zeros(col_zeros)
                _zeros[col] = self.binners[col].zeros
        return _zeros

    def fit_transform(self, df: pd.DataFrame, public: bool = False) -> pd.DataFrame:
        """
        Fit the binner to the DataFrame and transform it.

        :param df: The input DataFrame.
        :param public: Whether the data is public (no privacy concerns).
        :return: The transformed DataFrame.

        .. code-block:: python

            transformed_df = binner.fit_transform(df, public=True)
        """
        self.fit(df, public=public)
        t_df = self.transform(df)
        return t_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the DataFrame to its original form.

        :param df: The transformed DataFrame.
        :return: The original DataFrame.

        .. code-block:: python

            original_df = binner.inverse_transform(transformed_df)
        """
        dfs = [
            pd.Series(
                self.cat_encoders[col].inverse_transform(df[[col]]).squeeze(),
                index=df.index,
                name=col,
            )
            for col in self.cat_cols
        ]

        dfs += [
            pd.Series(
                self.binners[col].inverse_transform(df[col].to_numpy()),
                index=df.index,
                name=col,
            )
            for col in self.num_cols
        ]
        t_df = pd.concat(dfs, axis=1)

        # Add static columns
        for col, value in self.static_columns.items():
            t_df[col] = value

        # Apply NaN values
        for col, nan_col in self.nan_columns.items():
            nan_col = nan_col["name"]
            na_flag = t_df[nan_col].astype(bool)
            t_df.loc[na_flag, col] = np.nan
            t_df = t_df.drop(columns=[nan_col])

        t_df = t_df[self.col_order]

        return t_df.astype(self.dtypes)

    @to_path
    def store(self, path: Path) -> None:
        """
        Store the binner to a file.

        :param path: The file path to store the binner.

        .. code-block:: python

            binner.store("binner.pkl")
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        Load the binner from a file.

        :param path: The file path to load the binner from.
        :return: The loaded TableBinner instance.

        .. code-block:: python

            binner = TableBinner.load("binner.pkl")
        """
        return joblib.load(path)
