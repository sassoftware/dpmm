import pytest
import numpy as np
import pandas as pd
from dpmm.pipelines.aim import AIMPipeline
from dpmm.pipelines.priv_bayes import PrivBayesPipeline
from dpmm.pipelines.mst import MSTPipeline

@pytest.fixture
def sample_data():
    np.random.seed(2023)
    n_rows = 10_000
    data = {
        "int_col": np.random.randint(0, 100, n_rows),
        "float_col": np.random.rand(n_rows) * 100,
        "datetime_col": pd.date_range(start='1/1/2023', periods=n_rows, freq='D'),
        "timedelta_col": pd.to_timedelta(np.random.randint(1, 100, n_rows), unit='D'),
        "bool_col": np.random.choice([True, False], n_rows),
        "static_col": 'static_value',
        "categorical_col": pd.Categorical(np.random.choice(['A', 'B', 'C'], n_rows)),
        "categorical_numerical_col": pd.Categorical(np.random.choice(15, n_rows))
    }
    domain = {
        "int_col": {"lower": 0, "upper": 100},
        "float_col": {"lower": 0, "upper": 100},
        "datetime_col": {"lower": pd.Timestamp('2023-01-01'), "upper": pd.Timestamp('2023-01-10')},
        "timedelta_col": {"lower": pd.Timedelta(days=1), "upper": pd.Timedelta(days=99)},
        "bool_col": {"categories": [True, False]},
        "static_col": {"categories": ['static_value']},
        "categorical_col": {"categories": ['A', 'B', 'C']},
        "categorical_numerical_col": {"categories": list(range(15))}
    }
    return pd.DataFrame(data), domain

def validate_synthetic_data(synthetic_data, source_data, domain):
    assert synthetic_data.shape == source_data.shape
    for col, synth_series in synthetic_data.items():
        assert synth_series.dtype == source_data[col].dtype
        if synth_series.dtype.kind in 'Ob':
            assert pd.Series(synth_series.unique()).isin(domain[col]["categories"]).all()
        else:
            assert synth_series.min() >= domain[col]["lower"]
            assert synth_series.max() <= domain[col]["upper"]

def test_aim_Pipeline(sample_data):
    source_data, domain = sample_data
    Pipeline = AIMPipeline(epsilon=10, proc_epsilon=.1, gen_kwargs=dict(n_iters=10, rounds=1))
    Pipeline.fit(source_data, domain=domain)
    synthetic_data = Pipeline.generate(n_records=source_data.shape[0])
    validate_synthetic_data(synthetic_data, source_data, domain)

def test_priv_bayes_Pipeline(sample_data):
    source_data, domain = sample_data
    Pipeline = PrivBayesPipeline(epsilon=10, proc_epsilon=.1, gen_kwargs=dict(n_iters=10))
    Pipeline.fit(source_data, domain=domain)
    synthetic_data = Pipeline.generate(n_records=source_data.shape[0])
    validate_synthetic_data(synthetic_data, source_data, domain)

def test_mst_Pipeline(sample_data):
    source_data, domain = sample_data
    Pipeline = MSTPipeline(epsilon=10, proc_epsilon=.1, gen_kwargs=dict(n_iters=10))
    Pipeline.fit(source_data, domain=domain)
    synthetic_data = Pipeline.generate(n_records=source_data.shape[0])
    validate_synthetic_data(synthetic_data, source_data, domain)
