from dpmm.pipelines.base import MMPipeline
from dpmm.models.priv_bayes import PrivBayesGM


class PrivBayesPipeline(MMPipeline):
    model = PrivBayesGM
