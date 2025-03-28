import numpy as np
import pandas as pd

from dpmm.metrics.base import Metric


class PercentileSimilarity(Metric):
    metric_name = "percentile_similarity"

    def __init__(self, mode="max", step=1):
        self.mode = mode
        self.step = step

    def percentile_dist(self, source, target):
        source, target = source.astype(float), target.astype(float)
        _min, _max = min(source.min(), target.min()), max(source.max(), target.max())
        percentiles = np.arange(0, 100 + self.step, self.step)
        target_p = np.percentile(target, percentiles)
        source_p = np.percentile(source, percentiles)
        dist = (source_p - target_p) / (_max - _min)
        return dist

    def similarity(self, source, target):
        if self.mode == "max":
            agg = np.max
        elif self.mode == "mean":
            agg = np.mean
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        dist = self.percentile_dist(source, target)
        return 1 - agg(np.abs(dist))

    def __call__(
        self,
        synthetic_data: pd.DataFrame,
        source_data: pd.DataFrame,
        test_data: pd.DataFrame = None,
        domain: dict = None,
    ):
        return np.mean(
            [
                self.similarity(source=source_data[col], target=synthetic_data[col])
                for col, dtype in source_data.dtypes.items()
                if dtype.kind in "fui"
            ],
            axis=None,
        )
