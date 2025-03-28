from dpmm.engines.aim import AIMEngine
from dpmm.engines.mst import MSTEngine
from dpmm.engines.priv_bayes import PrivBayesEngine

ENGINES = [AIMEngine, MSTEngine, PrivBayesEngine]
ENGINE_DICT = {ENG.model.name: ENG for ENG in ENGINES}
