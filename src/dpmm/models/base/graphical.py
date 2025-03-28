import pandas as pd
from typing import Dict
from dpmm.models.base.base import GenerativeModel


class GraphicalGenerativeModel(GenerativeModel):
    def set_domain(self, domain: Dict):
        super().set_domain(domain)
        self.generator._domain = domain

    def set_structural_zeros(self, structural_zeros):
        self.generator.set_structural_zeros(structural_zeros)

    def check_fit(self, df):
        cls_name = self.__class__.__name__
        # check No Missing Values
        not_na = [col for col, series in df.items() if series.isna().any()]
        not_na_msg = (
            f"Columns {not_na} contains null values. Not supported by {cls_name}"
        )
        assert len(not_na) == 0, not_na_msg

        # Check All integers
        not_int = [col for col, series in df.items() if (series.dtype.kind not in "ui")]
        not_int_msg = (
            f"Columns {not_int} have non-int dtypes. Not Supported by {cls_name}"
        )
        assert len(not_int) == 0, not_int_msg

        # Check all Positive
        not_positive = [col for col, series in df.items() if (series < 0).any()]
        not_positive_msg = f"Columns {not_positive} have non positive values. Not Supported by {cls_name}"
        assert len(not_positive) == 0, not_positive_msg

        if getattr(self, "domain") is not None:
            upper = pd.Series(self.domain)
            real = df.max(axis=0)

            mismatch = real > upper

            if mismatch.any():
                to_high = real.loc[mismatch]
                raise ValueError(
                    f"Columns {to_high.index.tolist()} have values higher than provided domain."
                )
