import pytest
import numpy as np
import pandas as pd
from dpmm.processing.table_binner import TableBinner
from dpmm.processing.binners import PrivTreeBinner, UniformBinner
import logging


@pytest.fixture
def sample_dataframe():
    np.random.seed(2023)
    data = {
        "int_col": np.random.randint(0, 100, size=1000),
        "float_col": np.random.uniform(0, 1, size=1000),
        "cat_col": np.random.choice(["A", "B", "C"], size=1000),
        "datetime_col": pd.date_range("2023-01-01", periods=1000, freq="D"),
        "timedelta_col": pd.to_timedelta(
            np.random.randint(0, 100, size=1000), unit="D"
        ),
        "static_col": 1,
        "sparse_col": np.random.choice([np.nan, 0.5, 1.0], size=1000, p=[0.5, 0.3, 0.2]),
    }
    return pd.DataFrame(data)


@pytest.mark.parametrize("binner_type", ["uniform", "priv-tree"])
@pytest.mark.parametrize("use_domain", [True, False])
@pytest.mark.parametrize("public", [True, False])
@pytest.mark.parametrize("epsilon", [None, 10.0])
@pytest.mark.parametrize("n_bins", ["auto", 5, 10, 20])
@pytest.mark.parametrize("with_zeros", [True, False])
@pytest.mark.parametrize("serialise", [True, False])
def test_table_binner(
    tmp_path,
    binner_type,
    use_domain,
    public,
    epsilon,
    n_bins,
    with_zeros,
    serialise,
    caplog,
    sample_dataframe,
):
    caplog.set_level(logging.WARNING)

    domain = (
        {
            "int_col": {"lower": 0, "upper": 100},
            "float_col": {"lower": 0, "upper": 1},
            "cat_col": {"categories": ["A", "B", "C"]},
            "datetime_col": {
                "lower": pd.to_datetime("2023-01-01"),
                "upper": pd.to_datetime("2025-09-26"),
            },
            "timedelta_col": {
                "lower": pd.to_timedelta("0 days"),
                "upper": pd.to_timedelta("100 days"),
            },
            "sparse_col": {"lower": 0, "upper": 1},
            "static_col": {"lower": 0, "upper": 1},
        }
    )

    structural_zeros = None
    if with_zeros: 
        zero_cols = np.random.choice([col for col in domain.keys() if col != "sparse_col"], replace=False, size=3)
        structural_zeros = {}
        for col in zero_cols:
            dom = domain[col]
            if "categories" in dom:
                structural_zeros[col] = np.random.choice(dom["categories"], size=2, replace=False)
            else:
                zero_bounds = sorted(np.random.uniform(0, 1, size=6) * (dom["upper"] - dom["lower"]) + dom["lower"])
                structural_zeros[col] = [(zero_bounds[idx], zero_bounds[idx + 1]) for idx in range(0, 6, 2)]

    if not use_domain:
        domain = None

    binner_settings = {"n_bins": n_bins, "epsilon": epsilon}
    table_binner = TableBinner(
        binner_type=binner_type, binner_settings=binner_settings, domain=domain
    )

    if with_zeros:
        table_binner.set_structural_zeros(structural_zeros)

    if not use_domain:
        with caplog.at_level(logging.WARNING):
            table_binner.fit(sample_dataframe, public=public)
            if not public:
                assert (
                    "PrivacyLeakage: No categorical domain provided for Column cat_col - will be imputed."
                    in caplog.text
                )
    else:
        table_binner.fit(sample_dataframe, public=public)


    if with_zeros:
        for col, struct in structural_zeros.items():
            if sample_dataframe[col].dtype.kind in "Mmfui":
                for bin_idx in table_binner.zeros[col]:
                    assert any([table_binner.binners[col].bins[bin_idx] >= _s[0] and table_binner.binners[col].bins[bin_idx + 1] <= _s[1] for _s in struct])
            else:
                assert len(struct) == len(table_binner.zeros[col])

    transformed_df = table_binner.transform(sample_dataframe)
    assert not transformed_df.isnull().any(axis=None)

    if serialise:
        # Serialize
        file_path = tmp_path / "table_binner.pkl"
        table_binner.store(file_path)

        # Deserialize
        table_binner = TableBinner.load(file_path)
        loaded_transformed_df = table_binner.transform(sample_dataframe)

        pd.testing.assert_frame_equal(transformed_df, loaded_transformed_df)

    for col in sample_dataframe.select_dtypes(
        include=[np.number, np.datetime64, np.timedelta64]
    ).columns:
        if col not in transformed_df.columns:
            continue

        if isinstance(n_bins, int):
            assert transformed_df[col].nunique() <= n_bins
        else:
            assert transformed_df[col].nunique() <= table_binner.binners[col].n_bins
    
    for _, binner in table_binner.binners.items():
        if public or pd.isnull(epsilon):
            assert pd.isnull(binner.epsilon)
        else:
            assert binner.epsilon == (epsilon / len(table_binner.binners))

    inverse_transformed_df = table_binner.inverse_transform(transformed_df)
    for col in sample_dataframe.columns:
        if sample_dataframe[col].dtype.kind in "Mmfui":
            if use_domain:
                assert (
                    inverse_transformed_df[col]
                    .dropna()
                    .between(domain[col]["lower"], domain[col]["upper"])
                    .all()
                )
            elif epsilon is None:
                assert (
                    inverse_transformed_df[col]
                    .dropna()
                    .between(sample_dataframe[col].min(), sample_dataframe[col].max())
                    .all()
                )
            else:
                assert inverse_transformed_df[col].dtype == sample_dataframe[col].dtype
        else:
            if use_domain:
                assert set(inverse_transformed_df[col].unique()).issubset(
                    set(domain[col]["categories"])
                )
            else:
                assert set(inverse_transformed_df[col].unique()).issubset(
                    set(sample_dataframe[col].unique())
                )

        assert (
            inverse_transformed_df[col].isna() == sample_dataframe[col].isna()
        ).all()


    if with_zeros:
        for col, struct in structural_zeros.items():
            if sample_dataframe[col].dtype.kind in "Mmfui":
                not_zeros = ~(transformed_df[col].isin(table_binner.zeros[col]))
                nz_series = inverse_transformed_df[col].loc[not_zeros]

                for z in struct:
                    if nz_series.dtype.kind in "ui":
                        z = [int(_z) for _z in z]
                    
                    not_zero = nz_series.between(left=z[0], right=z[1], inclusive="neither")  
                    assert not_zero.sum() == 0
        

    # Check static column is preserved
    assert (inverse_transformed_df["static_col"] == 1).all()

    # Check sparsity is reproduced
    pd.testing.assert_series_equal(
        inverse_transformed_df["sparse_col"].isna(), sample_dataframe["sparse_col"].isna()
    )

    fit_transformed_df = table_binner.fit_transform(sample_dataframe)
    assert not fit_transformed_df.isnull().any().any()

    rnd = np.random.RandomState(42)
    table_binner.set_random_state(rnd)
    assert table_binner.random_state == rnd
    for binner in table_binner.binners.values():
        assert binner.rnd == rnd


@pytest.mark.parametrize("binner_type", ["uniform", "priv-tree"])
@pytest.mark.parametrize("use_domain", [True, False])
@pytest.mark.parametrize("epsilon", [None])
@pytest.mark.parametrize("n_bins", ["auto", 5, 10, 20])

def test_deterministic_behavior(
    binner_type, use_domain, epsilon, n_bins, sample_dataframe
):
    domain = (
        {
            "int_col": {"lower": 0, "upper": 100},
            "float_col": {"lower": 0, "upper": 1},
            "cat_col": {"categories": ["A", "B", "C"]},
            "datetime_col": {
                "lower": pd.to_datetime("2023-01-01"),
                "upper": pd.to_datetime("2025-09-26"),
            },
            "timedelta_col": {
                "lower": pd.to_timedelta("0 days"),
                "upper": pd.to_timedelta("100 days"),
            },
            "sparse_col": {"lower": 0, "upper": 1},
            "static_col": {"lower": 0, "upper": 1},
        }
        if use_domain
        else None
    )

    binner_settings = {"n_bins": n_bins, "epsilon": epsilon}
    rnd_seed = 42

    table_binner_1 = TableBinner(
        binner_type=binner_type, binner_settings=binner_settings, domain=domain
    )
    table_binner_1.set_random_state(np.random.RandomState(rnd_seed))
    table_binner_1.fit(sample_dataframe)
    transformed_df_1 = table_binner_1.transform(sample_dataframe)
    inverse_transformed_df_1 = table_binner_1.inverse_transform(transformed_df_1)

    table_binner_2 = TableBinner(
        binner_type=binner_type, binner_settings=binner_settings, domain=domain
    )
    table_binner_2.set_random_state(np.random.RandomState(rnd_seed))
    table_binner_2.fit(sample_dataframe)
    transformed_df_2 = table_binner_2.transform(sample_dataframe)
    inverse_transformed_df_2 = table_binner_2.inverse_transform(transformed_df_2)


    for col, binner in table_binner_1.binners.items():
        assert all(np.array(table_binner_2.binners[col].bins) == np.array(binner.bins))

    pd.testing.assert_frame_equal(transformed_df_1, transformed_df_2)
    pd.testing.assert_frame_equal(inverse_transformed_df_1, inverse_transformed_df_2)



