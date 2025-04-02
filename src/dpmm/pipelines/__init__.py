from dpmm.pipelines.aim import AIMPipeline
from dpmm.pipelines.mst import MSTPipeline
from dpmm.pipelines.priv_bayes import PrivBayesPipeline

PIPELINES = [AIMPipeline, MSTPipeline, PrivBayesPipeline]
PIPELINE_DICT = {PIPE.model.name: PIPE for PIPE in PIPELINES}
