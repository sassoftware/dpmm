from dpmm.engines.base import Engine
from dpmm.models.mst import MSTGM


class MSTEngine(Engine):
    model = MSTGM
