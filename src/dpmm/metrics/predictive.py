import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from dpmm.metrics.base import Metric


class PredError(Metric):
    metric_name = "predictive_error"

    def __init__(self, label, pred_model=None, *args, **kwargs):
        if pred_model is None:
            pred_model = LogisticRegression
            kwargs["n_jobs"] = 1
        self.pred_model = pred_model
        self.args = args
        self.kwargs = kwargs
        self.label = label

    def get_model(self):
        return self.pred_model(*self.args, **self.kwargs)

    def __call__(
        self,
        synthetic_data: pd.DataFrame,
        source_data: pd.DataFrame,
        test_data: pd.DataFrame = None,
        domain: dict = None,
    ):
        source_data, synthetic_data, test_data = self.normalise(
            source_data, synthetic_data, test_data
        )

        source_model = self.get_model()
        synth_model = self.get_model()

        # fit models : source model
        source_model.fit(
            source_data.drop(columns=[self.label]), source_data[self.label]
        )
        source_pred = source_model.predict(test_data.drop(columns=[self.label]))

        # fit models : synth model
        if synthetic_data[self.label].nunique() <= 1:
            synth_pred = np.ones(test_data.shape[0])
        else:
            synth_model.fit(
                synthetic_data.drop(columns=[self.label]), synthetic_data[self.label]
            )
            synth_pred = synth_model.predict(test_data.drop(columns=[self.label]))

        if test_data[self.label].nunique() <= 2:
            average = "binary"
        else:
            average = "weighted"
        synth_score = f1_score(
            y_true=test_data[self.label], y_pred=synth_pred, average=average
        )
        source_score = f1_score(
            y_true=test_data[self.label], y_pred=source_pred, average=average
        )

        del source_model, synth_model, source_pred, synth_pred
        return np.clip(synth_score / (source_score + 1e-6), 0, 1)
