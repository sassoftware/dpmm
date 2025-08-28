import pytest
import numpy as np
import pandas as pd
from dpmm.models import AIMGM, PrivBayesGM, MSTGM
from tempfile import TemporaryDirectory
from pathlib import Path


@pytest.fixture
def sample_dataframe():
    np.random.seed(2023)
    n_cols = 3
    columns = [f"col_{i}" for i in range(n_cols)]
    data = pd.DataFrame(np.random.randint(low=0, high=np.random.randint(10, 100), size=(10_000, n_cols)), columns=columns)
    return pd.DataFrame(data)


@pytest.mark.parametrize("model_class", [AIMGM, PrivBayesGM, MSTGM])
@pytest.mark.parametrize("use_domain", [True, False])
@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("max_model_size", [None, 80])
@pytest.mark.parametrize("epsilon", [None, 1])
@pytest.mark.parametrize("condition", [True, False])
@pytest.mark.parametrize("serialise", [True, False])
@pytest.mark.parametrize("with_zeros", [True, False])
@pytest.mark.parametrize("fit_mode", ["pretrain_only", "pretrain_and_fit", "fit_only"])
def test_models(model_class, use_domain, compress, max_model_size, epsilon, condition, serialise, sample_dataframe, with_zeros, fit_mode):
    if use_domain:
        domain =  {
            col: sample_dataframe[col].max() + 1
            for col in sample_dataframe.columns
        }
    else:
        domain = None


    structural_zeros = None
    if with_zeros:
        zero_col = np.random.choice(sample_dataframe.columns, size=1)[0]
        structural_zeros = {
            zero_col: np.random.choice(sample_dataframe[zero_col].max() + 1, replace=False, size=3)
        }

    random_state = np.random.RandomState(42)
    model_args = dict(
        epsilon=epsilon,
        domain=domain,
        compress=compress,
        max_model_size=max_model_size,
        n_iters=10,
    )

    if model_class.name == "aim":
        model_args["rounds"] = 5
    elif model_class.name == "priv-bayes":
        model_args["degree"] = 1
    
    model = model_class(**model_args)
    if with_zeros:
        model.set_structural_zeros(structural_zeros)

    model.set_random_state(random_state)
    # Test check_fit
    model.check_fit(sample_dataframe)

    # Test fit
    if fit_mode in ["pretrain_only", "pretrain_and_fit"]:
        model.fit(sample_dataframe, public=True)
        assert model.generator.cliques is not None
        assert model.generator.fit_state == "pretrained"

    
    if fit_mode in  ["fit_only", "pretrain_and_fit"]:
        model.fit(sample_dataframe)
        assert model.generator.fit_state == "trained"
    
    
        if max_model_size is not None:
            assert model.generator.model_size <= max_model_size
    
    

    if serialise:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Serialize
            model_path = tmp_path
            model.store(model_path)

            # Deserialize
            model = model_class.load(model_path)

    # Test generate

    if fit_mode in ["pretrain_only"]:
        err_msg = "Model has not been fully trained yet. please run a .fit call with `public` set to `False`"
        with pytest.raises(ValueError) as e:
            synthetic_data = model.generate(n_records=100)
            assert err_msg in str(e.value)
    else:
        if condition:
            condition_cols = np.random.choice(sample_dataframe.columns, size=1, replace=False)
            condition_df = sample_dataframe[condition_cols]
            if with_zeros: 
                for col, _z in structural_zeros.items():
                    if col in condition_df.columns:
                        condition_df = condition_df.loc[~condition_df[col].isin(_z)]

            synthetic_data = model.generate(condition_records=condition_df)
            pd.testing.assert_frame_equal(condition_df, synthetic_data[condition_cols])
        else:
            synthetic_data = model.generate(n_records=100)
            assert synthetic_data.shape[0] == 100

        # Check synthetic data range
        for col in sample_dataframe.columns:
            assert synthetic_data[col].min() >= sample_dataframe[col].min()
            assert synthetic_data[col].max() <= sample_dataframe[col].max()

        if with_zeros:
            is_zero = synthetic_data[zero_col].isin(structural_zeros[zero_col])
            assert is_zero.sum() == 0

# @pytest.mark.parametrize("model_class", [AIMGM, PrivBayesGM, MSTGM])
# def test_deterministic_behavior(model_class, sample_dataframe):
#     random_state = np.random.RandomState(42)
#     model_1 = model_class()
#     model_1.set_random_state(random_state)
#     model_1.fit(sample_dataframe)
#     synthetic_data_1 = model_1.generate(n_records=100)


#     random_state = np.random.RandomState(42)
#     model_2 = model_class()
#     model_2.set_random_state(random_state)
#     model_2.fit(sample_dataframe)
#     synthetic_data_2 = model_2.generate(n_records=100)

#     pd.testing.assert_frame_equal(synthetic_data_1, synthetic_data_2)