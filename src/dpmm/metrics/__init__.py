from dpmm.metrics.corr import CorrelationError
from dpmm.metrics.discriminator import DiscriminatorSimilarity
from dpmm.metrics.ks import KSTest
from dpmm.metrics.marginal import MarginalSimilarity
from dpmm.metrics.mutual_info import MutualInformationSimilarity
from dpmm.metrics.percentile import PercentileSimilarity
from dpmm.metrics.predictive import PredError
from dpmm.metrics.query import QuerySimilarity

METRICS = [
    CorrelationError,
    MarginalSimilarity,
    KSTest,
    DiscriminatorSimilarity,
    MutualInformationSimilarity,
    PercentileSimilarity,
    PredError,
    QuerySimilarity,
]

METRIC_DICT = {metric.metric_name: metric for metric in METRICS}
