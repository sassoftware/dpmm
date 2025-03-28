from logging import getLogger
from pathlib import Path
from typing import Dict, List, Self, Tuple

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
        binner_settings={"n_bins": "auto"},
        domain=None,
        structural_zeros: Dict[str, Tuple] = None,
        random_state=None,
    ):
        # Initialize TableBinner with settings, domain, and random state
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
        self.is_fit = False

        if structural_zeros is None:
            structural_zeros = {}
        self.set_structural_zeros(structural_zeros)

    def set_params(self, **kwargs):
        # Update binner settings with provided parameters
        self.binner_settings.update(kwargs)

    def set_domain(self, domain: Dict):
        # Set the domain for the binner
        if domain is not None:
            self.domain = domain

    def set_structural_zeros(self, structural_zeros: Dict[str, Tuple]):
        # Set the structural zeros for the binner
        self.structural_zeros = structural_zeros
        if self.binners is not None:
            for col, binner in self.binners.items():
                if col in structural_zeros:
                    binner.set_structural_zeros(structural_zeros[col])

    def set_random_state(self, rnd: RandomState):
        # Set the random state for reproducibility
        self.random_state = rnd

        if self.binners is not None:
            for _, binner in self.binners.items():
                binner.set_random_state(rnd)

    def get_categories(
        self, col: str, series: pd.Series, public: bool = False
    ) -> List[str]:
        # Get categories for a column, handling missing domain information
        col_domain = self.domain.get(col, {})
        if "categories" in col_domain:
            categories = col_domain["categories"]
        else:
            if not (public):
                logger.warning(
                    f"PrivacyLeakage: No categorical domain provided for Column {col} - will be imputed."
                )
            categories = series.unique()

        all_numerical = all(
            [
                (
                    np.issubdtype(np.dtype(type(cat)), np.floating)
                    or np.issubdtype(np.dtype(type(cat)), np.bool_)
                )
                for cat in categories
                if pd.notnull(cat)
            ]
        )

        if all_numerical:
            categories = sorted(categories, key=lambda x: (pd.isnull(x), x))

        return categories

    def insert_col(self, df, col, series):
        # Insert a new column into the DataFrame, ensuring no name conflicts
        idx = 0
        while col in df.columns:
            col = f"{col}_{idx}"
            idx += 1

        return pd.concat([df, series.rename(col)], axis=1), col

    def fit(self, df: pd.DataFrame, public=False):
        # Fit the binner to the DataFrame
        self.dtypes = df.dtypes
        self.binners = {}
        self.cat_encoders = {}
        self.bin_domain = {}
        self.nan_columns = {}
        self.static_columns = {}
        self.col_order = df.columns.tolist()

        # NaN Management
        for col, series in df.items():
            na_flag = series.isna()
            not_na_flag = ~na_flag
            if (series.dtype.kind in "Mmfui") and (na_flag.any() and not_na_flag.any()):
                fill_value = (
                    df.loc[not_na_flag, col]
                    .sample(n=1, random_state=self.random_state)
                    .iloc[0]
                )
                df, nan_col = self.insert_col(
                    df=df, col=f"{col}_NaN", series=na_flag.astype("category")
                )
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
            col for col, series in df.items() if series.dtype.kind in "Mmfui"
        ]

        # Compute Epsilon
        if public:
            epsilon = None
        else:
            epsilon = self.binner_settings.get("epsilon", None)
            if epsilon is not None:
                # Split the epsilon
                epsilon /= len(self.num_cols)

        # Categorical Columns
        self.cat_cols = [col for col in df.columns if col not in self.num_cols]
        for col in self.cat_cols:
            categories = self.get_categories(col, df[col], public=public)
            self.bin_domain[col] = len(categories)
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
        # Transform the DataFrame using the fitted binners and encoders

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
    def n_bins(self):
        # Return the number of bins for each column
        return {
            col: binner.n_bins
            for col, binner in self.binners.items()
            if not isinstance(binner, dict)
        }

    @property
    def spent_epsilon(self):
        # Return the spent epsilon for each column
        return {
            col: binner.spent_epsilon
            for col, binner in self.binners.items()
            if not isinstance(binner, dict)
        }

    @property
    def zeros(self):
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

    def fit_transform(self, df: pd.DataFrame, public=False) -> pd.DataFrame:
        # Fit and transform the DataFrame
        self.fit(df, public=public)
        t_df = self.transform(df)
        return t_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Inverse transform the DataFrame to its original form
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
    def store(self, path: Path):
        # Store the binner to a file
        joblib.dump(self, path)

    @classmethod
    def load(self, path: Path) -> Self:
        # Load the binner from a file
        return joblib.load(path)
