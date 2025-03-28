import numpy as np
import pandas as pd
from scipy.stats import kstest

from dpmm.metrics.base import Metric


class KSTest(Metric):
    metric_name = "ks_test"

    def ks(self, source: pd.Series, target: pd.Series):
        return kstest(source, target)[1]

    def __call__(
        self,
        synthetic_data: pd.DataFrame,
        source_data: pd.DataFrame,
        test_data: pd.DataFrame = None,
        domain: dict = None,
    ):
        return np.mean(
            [
                self.ks(source=source_data[col], target=synthetic_data[col])
                for col, dtype in source_data.dtypes.items()
                if dtype.kind in "fui"
            ],
            axis=None,
        )
