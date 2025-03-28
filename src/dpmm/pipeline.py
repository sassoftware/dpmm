from pathlib import Path
from typing import Dict, Self, Union

import pandas as pd
from numpy.random import RandomState

from dpmm.models import load_model
from dpmm.models.base.base import GenerativeModel
from dpmm.processing.table_binner import TableBinner
from dpmm.utils import to_path


class GenerativePipeline:
    def __init__(self, gen: GenerativeModel, proc: TableBinner = None):
        self.gen = gen
        self.proc = proc
        self.random_state = None

    def set_random_state(self, rnd: Union[int, RandomState] = None):
        if not isinstance(rnd, RandomState):
            rnd = RandomState(rnd)

        self.random_state = rnd
        if self.proc is not None:
            self.proc.set_random_state(rnd)
        self.gen.set_random_state(rnd)

    def fit(
        self,
        df: pd.DataFrame,
        domain: Dict = None,
        structural_zeros: Dict = None,
        random_state: Union[int, RandomState] = None,
        public=False,
    ):
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
        self.gen.fit(t_df, public=public)
        

    def generate(
        self,
        n_records: int = None,
        condition_records: pd.DataFrame = None,
        random_state: Union[int, RandomState] = None,
    ) -> pd.DataFrame:
        msg = "InvalidInput: Either 'n' or 'condition_records' must be set, both provided as None."
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
    def store(self, path: Path):
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
        proc_path = path / "processing.joblib"
        if proc_path.exists():
            proc = TableBinner.load(proc_path)
        else:
            proc = None

        gen = load_model(path / "generative_model")
        return cls(gen=gen, proc=proc)
