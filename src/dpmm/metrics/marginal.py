import numpy as np
import pandas as pd

from dpmm.metrics.base import Metric


class MarginalSimilarity(Metric):
    metric_name = "marginal_similarity"

    def __init__(self, n_bins=50):
        self.n_bins = n_bins

    def marginal_histogram(self, data: pd.Series, bins: list, is_category=False):
        if is_category:
            hist = data.value_counts(normalize=True).reindex(bins, fill_value=0)
        else:
            hist = np.histogram(data, bins=bins)[0]
            hist = hist.astype(float) / hist.sum()

        hist = pd.Series(hist).fillna(0)
        return hist

    def marginal_similarity(
        self, synthetic_series: pd.DataFrame, source_series: pd.DataFrame, domain: dict
    ):
        if domain is None:
            if source_series.dtype.kind in "Osb":
                is_category = True
                domain = {"categories": source_series.unique().tolist()}
            else:
                is_category = False
                domain = {"lower": source_series.min(), "upper": source_series.max()}

        if "categories" in domain:
            is_category = True
            bins = domain["categories"]
        else:
            is_category = False
            bins = np.linspace(
                domain["lower"],
                domain["upper"],
                self.n_bins + 1,
            )

        synthetic_hist = self.marginal_histogram(synthetic_series, bins, is_category)
        source_hist = self.marginal_histogram(source_series, bins, is_category)

        return np.minimum(synthetic_hist, source_hist).sum()

    def __call__(self, synthetic_data, source_data, test_data=None, domain=None):
        if domain is None:
            domain = {}

        return np.mean(
            [
                self.marginal_similarity(
                    synthetic_data[col], source_data[col], domain.get(col, None)
                )
                for col in source_data.columns
            ]
        )
