import json
from numpy.random import RandomState
from pathlib import Path
from typing import Self, Dict
import pandas as pd


class GenerativeModel:
    "Base class for all generative models"
    name: str = None

    def __init__(self, domain: Dict, random_state: RandomState = None):
        self.domain = domain
        self.random_state = random_state

    def set_domain(self, domain: Dict):
        self.domain = domain

    def set_random_state(self, random_state: RandomState):
        self.random_state = random_state

    def fit(self, df: pd.DataFrame):
        "Fit a generative model"
        return NotImplementedError("Method needs to be overwritten.")

    def check_fit(self, df: pd.DataFrame):
        "Fit a generative model"
        return NotImplementedError("Method needs to be overwritten.")

    def generate(self, n_records: int = None, condition_records: pd.DataFrame = None):
        "Generate a synthetic dataset of size n_records"
        return NotImplementedError("Method needs to be overwritten.")

    def store(self, path: Path):
        "Generate a synthetic dataset of size n_records"
        with (path / "model_type.json").open("w") as fw:
            json.dump(self.name, fw)

    @classmethod
    def load(cls, path: Path) -> Self:
        "Generate a synthetic dataset of size n_records"
        return NotImplementedError("Method needs to be overwritten.")
