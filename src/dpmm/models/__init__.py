import json
from pathlib import Path

from dpmm.models.aim import AIMGM
from dpmm.models.base.base import GenerativeModel
from dpmm.models.mst import MSTGM
from dpmm.models.priv_bayes import PrivBayesGM

MODELS = [PrivBayesGM, AIMGM, MSTGM]
MODEL_DICT = {MODEL.name: MODEL for MODEL in MODELS}


def load_model(path: Path) -> GenerativeModel:
    with (path / "model_type.json").open("r") as fr:
        model_type = json.load(fr)

    return MODEL_DICT[model_type].load(path)
