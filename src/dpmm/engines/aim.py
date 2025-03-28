from dpmm.engines.base import Engine
from dpmm.models.aim import AIMGM


class AIMEngine(Engine):
    model = AIMGM
