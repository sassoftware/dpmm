from dpmm.pipelines.base import MMPipeline
from dpmm.models.mst import MSTGM


class MSTPipeline(MMPipeline):
    model = MSTGM
