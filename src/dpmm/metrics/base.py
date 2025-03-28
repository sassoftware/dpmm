import pandas as pd


class Metric:
    metric_name = "base_metric"

    def __call__(
        self,
        synthetic_data: pd.DataFrame,
        source_data: pd.DataFrame,
        test_data: pd.DataFrame = None,
        domain: dict = None,
    ):
        # Placeholder for metric calculation logic
        # This could include statistical tests, visualizations, etc.
        raise NotImplementedError("Metric calculation not implemented.")

    def normalise(
        self,
        synthetic_data: pd.DataFrame,
        source_data: pd.DataFrame,
        test_data: pd.DataFrame = None,
        domain: dict = None,
    ):
        source_data, synthetic_data = source_data.copy(), synthetic_data.copy()
        if test_data is not None:
            test_data = test_data.copy()
        # Normalise data
        for col, series in source_data.items():
            if series.dtype.kind in "Osb":
                vmap = {cat: idx for idx, cat in enumerate(series.unique())}
                source_data[col], synthetic_data[col] = source_data[col].map(
                    vmap
                ), synthetic_data[col].map(vmap)
                source_data[col], synthetic_data[col] = source_data[col].astype(
                    "object"
                ).fillna(len(vmap)), synthetic_data[col].astype("object").fillna(
                    len(vmap)
                )
                if test_data is not None:
                    test_data[col] = (
                        test_data[col].map(vmap).astype("object").fillna(len(vmap))
                    )
        return source_data, synthetic_data, test_data
