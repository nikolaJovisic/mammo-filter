from enum import Enum

class Aggregation(Enum):
    MEAN = "mean"
    MAX = "max"
    
def aggregate(logits, group, aggregation: Aggragation):
    if aggregation == Aggragation.MEAN:
        return scatter_mean(logits, group, dim=0)
    if aggragation == Aggragation.MAX:
        max_logits, _ = scatter_max(logits, group, dim=0)
        return max_logits
    raise ValueError("Aggregation not supported!")