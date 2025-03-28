from collections import defaultdict

import numpy as np
import pandas as pd

from dpmm.metrics.base import Metric
from dpmm.models.priv_bayes import MI, Dataset, Domain


class MutualInformationSimilarity(Metric):
    metric_name = "mutual_information_similarity"

    def __init__(self, n_bins=50, eps=1e-5):
        self.n_bins = n_bins
        self.eps = eps

    def discretise(self, data: pd.DataFrame, domain: dict):
        "returns data where all columns are integer mapping for categorical data and discretising into uniform bins for continuous data"
        data = data.copy()
        for col, series in data.items():
            if "categories" in domain[col]:
                vmap = {cat: idx for idx, cat in enumerate(domain[col]["categories"])}
                data[col] = (
                    data[col].map(vmap).astype("object").fillna(len(vmap)).astype(int)
                )
            else:
                bins = np.linspace(
                    domain[col]["lower"],
                    domain[col]["upper"],
                    self.n_bins + 1,
                )
                data[col] = np.digitize(data[col], bins)
                data[col].fillna(self.n_bins, inplace=True)

        discrete_domain = data.max(axis=0).to_dict()
        return Dataset(df=data, domain=Domain.fromdict(discrete_domain))

    def __call__(
        self,
        synthetic_data: pd.DataFrame,
        source_data: pd.DataFrame,
        test_data: pd.DataFrame = None,
        domain: dict = None,
    ):
        if domain is None:
            domain = {
                col: (
                    {"lower": source_data[col].min(), "upper": source_data[col].max()}
                    if series.dtype.kind not in "Osb"
                    else {"categories": source_data[col].unique().tolist()}
                )
                for col, series in source_data.items()
            }

        source_data, synthetic_data, _ = self.normalise(source_data, synthetic_data)
        source_dataset = self.discretise(source_data, domain)
        synthetic_dataset = self.discretise(synthetic_data, domain)

        source_MI = MI(source_dataset)
        synth_MI = MI(synthetic_dataset)

        scores = defaultdict(dict)
        columns = source_data.columns.tolist()

        for idx, col_1 in enumerate(columns):
            for col_2 in columns[idx + 1 :]:
                source_score = source_MI(child=col_1, parents=(col_2,))
                synth_score = synth_MI(child=col_1, parents=(col_2,))
                sim_score = min(source_score, synth_score) / (
                    max(source_score, synth_score) + self.eps
                )

                scores[col_1][col_2] = sim_score

        return np.mean(
            [score for score_dict in scores.values() for score in score_dict.values()]
        )
