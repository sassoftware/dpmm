from dpmm.pipelines.base import MMPipeline
from dpmm.models.aim import AIMGM


class AIMPipeline(MMPipeline):
    model = AIMGM
