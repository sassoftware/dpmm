import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from dpmm.metrics.base import Metric

# This metric evaluates the similarity between the discriminator's predictions on synthetic and source data.


class DiscriminatorSimilarity(Metric):
    metric_name = "discriminator_similarity"

    def __init__(self, test_size=0.2):
        self.test_size = test_size

    def __call__(
        self,
        synthetic_data: pd.DataFrame,
        source_data: pd.DataFrame,
        test_data: pd.DataFrame = None,
        domain: dict = None,
    ):
        source_data, synthetic_data, _ = self.normalise(
            source_data, synthetic_data, test_data
        )

        clf = LogisticRegression()
        X, y = pd.concat(
            [source_data, synthetic_data], axis=0, ignore_index=False
        ), pd.Series([1] * source_data.shape[0] + [0] * synthetic_data.shape[0])
        train_X, test_X, train_y, _ = train_test_split(
            X, y, test_size=self.test_size, stratify=y
        )
        clf.fit(train_X, train_y)
        y_pred = clf.predict_proba(test_X)
        score = 1 - (2 * np.abs(0.5 - y_pred[:, 0]).mean())

        del y_pred, X, y, clf, train_X, train_y, test_X
        return score
