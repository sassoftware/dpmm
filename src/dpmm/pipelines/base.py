from pathlib import Path
from typing import Dict, Optional, Self, Union

import pandas as pd
from numpy.random import RandomState

from dpmm.models import load_model
from dpmm.models.base.base import GenerativeModel
from dpmm.processing.table_binner import TableBinner
from dpmm.utils import to_path


class GenerativePipeline:
    def __init__(self, gen: GenerativeModel, proc: Optional[TableBinner] = None):
        """
        Initialize A GenerativePipeline object.
        This object is used to chain together a generative model and a preprocessing step.
        Ensuring that generative model can ingest any data type [e.g. categorical, numerical, etc.]
        This will also ensure that the domain and structural zeros are set correctly for the generative model.

        :param gen: The generative model.
        :param proc: Optional table binner for preprocessing.

        .. code-block:: python
            from dpmm.models.mst import MSTGM
            from dpmm.processing.table_binner import TableBinner
            model = MSTGM(epsilon = 1.0, delta = 1e-5)
            binner = TableBinner()
            pipeline = GenerativePipeline(gen=model, proc=binner)
        """
        self.gen = gen
        self.proc = proc
        self.random_state = None

    def set_random_state(self, rnd: Union[int, RandomState] = None) -> None:
        """
        Set the random state for reproducibility.

        :param rnd: An integer seed or a RandomState instance.

        .. code-block:: python

            pipeline.set_random_state(42)
        """
        if not isinstance(rnd, RandomState):
            rnd = RandomState(rnd)

        self.random_state = rnd
        if self.proc is not None:
            self.proc.set_random_state(rnd)
        self.gen.set_random_state(rnd)

    def fit(
        self,
        df: pd.DataFrame,
        domain: Optional[Dict] = None,
        structural_zeros: Optional[Dict] = None,
        random_state: Union[int, RandomState] = None,
        public: bool = False,
        marginals_only: bool = False,
    ) -> None:
        """
        Fit the pipeline to the data.

        :param df: The input DataFrame.
        :param domain: Optional domain information for the data.
        :param structural_zeros: Optional structural zeros for specific columns.
        :param random_state: An integer seed or a RandomState instance.
        :param public: Whether the data is public (no privacy concerns).

        .. code-block:: python

            pipeline.fit(df, domain={"col1": {"lower": 0, "upper": 10}})
        """
        self.set_random_state(random_state)
        # Processing
        if self.proc is not None:
            if not (self.proc.is_fit):
                self.proc.set_domain(domain)
                t_df = self.proc.fit_transform(df, public=public)
            else:
                t_df = self.proc.transform(df)

            if structural_zeros is not None:
                self.proc.set_structural_zeros(structural_zeros)
            zeros = self.proc.zeros
            t_domain = self.proc.bin_domain
        else:
            t_domain = domain
            t_df = df
            zeros = structural_zeros

        # Generation
        self.gen.set_domain(domain=t_domain)
        self.gen.check_fit(t_df)
        if zeros is not None:
            self.gen.set_structural_zeros(zeros)
        self.gen.fit(t_df, public=public, marginals_only=marginals_only)

    def generate(
        self,
        n_records: Optional[int] = None,
        condition_records: Optional[pd.DataFrame] = None,
        random_state: Union[int, RandomState] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic data.

        :param n_records: Number of records to generate.
        :param condition_records: Optional DataFrame to condition the generation.
        :param random_state: An integer seed or a RandomState instance.
        :return: A DataFrame of synthetic data.

        .. code-block:: python

            synthetic_data = pipeline.generate(n_records=100)
        """
        msg = "InvalidInput: Either 'n_records' or 'condition_records' must be set, both provided as None."
        assert (condition_records is not None) or (n_records is not None), msg

        if condition_records is not None:
            n_records = condition_records.shape[0]
            t_condition = self.proc.transform(condition_records)
        else:
            t_condition = None
        self.set_random_state(random_state)
        synth_df = self.gen.generate(n_records=n_records, condition_records=t_condition)

        if self.proc is not None:
            synth_df = self.proc.inverse_transform(df=synth_df)

        if condition_records is not None:
            synth_df = pd.concat(
                [condition_records, synth_df.drop(columns=condition_records.columns)],
                axis=1,
            )
        return synth_df

    @to_path
    def store(self, path: Path) -> None:
        """
        Store the pipeline to a file.

        :param path: The file path to store the pipeline.

        .. code-block:: python

            pipeline.store("pipeline_path")
        """
        # Processing
        if self.proc is not None:
            proc_path = path / "processing.joblib"
            self.proc.store(proc_path)

        # Generation
        gen_path = path / "generative_model"
        gen_path.mkdir(exist_ok=True, parents=True)
        self.gen.store(gen_path)

    @classmethod
    @to_path
    def load(cls, path: Path) -> Self:
        """
        Load the pipeline from a file.

        :param path: The file path to load the pipeline from.
        :return: The loaded GenerativePipeline instance.

        .. code-block:: python

            pipeline = GenerativePipeline.load("pipeline_path")
        """
        proc_path = path / "processing.joblib"
        if proc_path.exists():
            proc = TableBinner.load(proc_path)
        else:
            proc = None

        gen = load_model(path / "generative_model")
        return GenerativePipeline(gen=gen, proc=proc)


class MMPipeline(GenerativePipeline):
    model: GenerativeModel = None

    def __init__(
        self,
        # Model Params
        epsilon: float = 1,
        delta: float = 1e-5,
        compress: bool = True,
        max_model_size: Optional[int] = None,
        n_jobs: int = -1,
        gen_kwargs: Optional[Dict] = None,
        # Processing Params
        binner_type: Optional[str] = None,
        proc_epsilon: float = 0.1,
        n_bins: Union[str, int] = "auto",
        binner_kwargs: Optional[Dict] = None,
        disable_processing: bool = False,
    ):
        """
        Initialize the MMPipeline.

        :param epsilon: Privacy budget for the generative model.
        :param delta: Delta parameter for differential privacy.
        :param gen_kwargs: Additional arguments for the generative model.
        :param binner_type: Type of binner to use (e.g., "priv-tree").
        :param proc_epsilon: Privacy budget for the processing step.
        :param n_bins: Number of bins or 'auto' for automatic binning.
        :param binner_kwargs: Additional arguments for the binner.
        :param disable_processing: Whether to disable the processing step.

        .. code-block:: python

            pipeline = MMPipeline(
                epsilon=1.0,
                delta=1e-5,
                binner_type="uniform",
                proc_epsilon=0.1,
                n_bins=10,
            )
        """
        if gen_kwargs is None:
            gen_kwargs = {}
        if binner_kwargs is None:
            binner_kwargs = {}

        if binner_type is None:
            if proc_epsilon is not None:
                binner_type = "priv-tree"
            else:
                binner_type = "uniform"

        gen = self.model(
            epsilon=epsilon,
            delta=delta,
            compress=compress,
            max_model_size=max_model_size,
            n_jobs=n_jobs,
            **gen_kwargs
        )

        if disable_processing:
            proc = None
        else:
            proc = TableBinner(
                binner_type=binner_type,
                binner_settings=dict(
                    epsilon=proc_epsilon, n_bins=n_bins, **binner_kwargs
                ),
            )

        super().__init__(gen=gen, proc=proc)

    @property
    def model_size(self):
        return self.gen.model_size
