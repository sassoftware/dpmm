import pytest
import numpy as np
import pandas as pd
from dpmm.models import AIMGM, PrivBayesGM, MSTGM
from dpmm.pipelines.base import GenerativePipeline
from dpmm.processing.table_binner import TableBinner
from tempfile import TemporaryDirectory
from pathlib import Path

@pytest.fixture
def sample_dataframe():
    np.random.seed(2023)
    n_rows = 10_000
    data = {
        "int_col": np.random.randint(0, 100, n_rows),
        "float_col": np.random.rand(n_rows) * 100,
        "datetime_col": pd.date_range(start='1/1/2023', periods=n_rows, freq='D'),
        "timedelta_col": pd.to_timedelta(np.random.randint(1, 100, n_rows), unit='D'),
        "bool_col": np.random.choice([True, False], n_rows),
        "static_col": 'static_value',
        "categorical_col": pd.Categorical(np.random.choice(['A', 'B', 'C'], n_rows))
    }
    return pd.DataFrame(data)

@pytest.mark.parametrize("model_class", [AIMGM, PrivBayesGM, MSTGM])
@pytest.mark.parametrize("use_domain", [True, False])
@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("with_zeros", [True, False])
@pytest.mark.parametrize("condition", [True, False])
@pytest.mark.parametrize("max_model_size", [None, 80])
@pytest.mark.parametrize("epsilon", [None, 1])
@pytest.mark.parametrize("serialise", [True, False])
@pytest.mark.parametrize("fit_mode", ["pretrain_only", "pretrain_and_fit", "fit_only"])
def test_pipeline(model_class, use_domain, compress, with_zeros, condition, max_model_size, epsilon, serialise, sample_dataframe, fit_mode):
    if use_domain:
        domain =  {
            col: ({
                "lower": sample_dataframe[col].min(),
                "upper": sample_dataframe[col].max()}
                if sample_dataframe[col].dtype.kind not in "Ob"
                else {"categories": sample_dataframe[col].unique().tolist()})
            
            for col in sample_dataframe.columns if sample_dataframe[col].dtype != 'object'
        }
    else:
        domain = None

    structural_zeros = None
    if with_zeros: 
        zero_cols = np.random.choice([col for col in sample_dataframe.columns if col != "sparse_col"], replace=False, size=3)
        structural_zeros = {}
        for col in zero_cols:
            if sample_dataframe[col].dtype.kind in "Ob":
                structural_zeros[col] = np.random.choice(sample_dataframe[col].unique(), size=2, replace=False)
            else:
                upper, lower = sample_dataframe[col].max(), sample_dataframe[col].min()
                zero_bounds = sorted(np.random.uniform(0, 1, size=6) * (upper - lower) + lower)
                structural_zeros[col] = [(zero_bounds[idx], zero_bounds[idx + 1]) for idx in range(0, 6, 2)]

    random_state = np.random.RandomState(42)
    model_args = dict(
        epsilon=epsilon,
        domain=domain,
        compress=compress,
        max_model_size=max_model_size,
        n_iters=10,
    )

    if isinstance(model_class, AIMGM):
        model_args["rounds"] = 1
    
    model = model_class(**model_args)
    binner = TableBinner()
    pipeline = GenerativePipeline(gen=model, proc=binner)
    pipeline.set_random_state(random_state)
    

    # Test fit
    if fit_mode in ["pretrain_only", "pretrain_and_fit"]:
        pipeline.fit(sample_dataframe, domain=domain, public=True, structural_zeros=structural_zeros)
        assert pipeline.gen.generator.cliques is not None

    if fit_mode in ["fit_only", "pretrain_and_fit"]:
        pipeline.fit(sample_dataframe, domain=domain, structural_zeros=structural_zeros)

    if serialise:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Serialize
            pipeline.store(tmp_path)

            # Deserialize
            pipeline = GenerativePipeline.load(tmp_path)

    # Test generate
    if fit_mode in ["pretrain_only"]:
        err_msg = "Model has not been fully trained yet. please run a .fit call with `public` set to `False`"
        with pytest.raises(ValueError) as e:
            synthetic_data = pipeline.generate(n_records=100)
            assert err_msg in str(e.value)
    else:
        if condition:
            condition_cols = np.random.choice(sample_dataframe.columns, size=2, replace=False)
            condition_df = sample_dataframe[condition_cols]
            if with_zeros:
                for col, _z in structural_zeros.items():
                    if col in condition_df.columns:
                        if condition_df[col].dtype.kind in "ObS":
                            condition_df = condition_df.loc[~condition_df[col].isin(_z)]
                        else:
                            for z in _z:
                                condition_df = condition_df.loc[~(condition_df[col].between(*z))]
            synthetic_data = pipeline.generate(condition_records=condition_df)
            pd.testing.assert_frame_equal(condition_df, synthetic_data[condition_cols])
        else:
            synthetic_data = pipeline.generate(n_records=100)
            assert synthetic_data.shape[0] == 100

        # Check synthetic data range
        for col in sample_dataframe.columns:
            if sample_dataframe[col].dtype.kind not in "Ob":
                assert synthetic_data[col].min() >= sample_dataframe[col].min()
                assert synthetic_data[col].max() <= sample_dataframe[col].max()
            else:
                assert synthetic_data[col].isin(sample_dataframe[col].unique()).all()

        
        if with_zeros:
            for col, struct in structural_zeros.items():
                if synthetic_data[col].dtype.kind in "Mmfui":
                    for z in struct:
                        assert synthetic_data[col].between(*z).sum() == 0
                else:
                    assert not synthetic_data[col].isin(struct).any()
