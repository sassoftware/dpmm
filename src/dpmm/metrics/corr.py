import numpy as np
import pandas as pd

from dpmm.metrics.base import Metric


class CorrelationError(Metric):
    metric_name = "correlation_error"

    def __call__(
        self,
        synthetic_data: pd.DataFrame,
        source_data: pd.DataFrame,
        test_data: pd.DataFrame = None,
        domain: dict = None,
    ):
        num_cols = [
            col for col, dtype in source_data.dtypes.items() if (dtype.kind in "fui")
        ]
        synthetic_data, source_data = synthetic_data[num_cols], source_data[num_cols]
        synth_corr, source_corr = synthetic_data.corr(), source_data.corr()
        min_corr = min((synth_corr).min(axis=None), (source_corr).min(axis=None))
        synth_corr, source_corr = (synth_corr - min_corr), (source_corr - min_corr)
        score = (
            np.minimum(synth_corr, source_corr)
            / (np.maximum(synth_corr, source_corr) + 1e-5)
        ).mean(axis=None)
        del min_corr, synth_corr, source_corr
        return score
