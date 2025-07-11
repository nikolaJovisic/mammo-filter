from enum import Enum
from torch_scatter import scatter_mean, scatter_max

class Aggregation(Enum):
    MEAN = "mean"
    MAX = "max"
    
def aggregate(logits, group, aggregation: Aggregation):
    if aggregation == Aggregation.MEAN or aggregation is None:
        return scatter_mean(logits, group, dim=0)
    if aggregation == Aggregation.MAX:
        max_logits, _ = scatter_max(logits, group, dim=0)
        return max_logits
    raise ValueError("Aggregation not supported!")