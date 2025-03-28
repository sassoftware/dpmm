from dpmm.engines.base import Engine
from dpmm.models.priv_bayes import PrivBayesGM


class PrivBayesEngine(Engine):
    model = PrivBayesGM
