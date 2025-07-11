import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.aggregation import aggregate, Aggregation
from utils.collate import collate

def _evaluate_one_agg(model, dataset, batch_size, aggregation, device):
    dataset = collate(dataset, batch_size)
    if aggregation is None:
        raise ValueError("No sense in not aggregating when testing!")
    
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y, _, group in dataset:
            x, group = x.to(device), group.to(device)
            logits = model(x)

            logits = logits[group >= 0]
            group = group[group >= 0]

            aggragated_logits = aggregate(logits, group, aggregation)

            all_logits.append(aggragated_logits.cpu())
            all_labels.append(y)

    logits = torch.cat(all_logits).squeeze(1)
    labels = torch.cat(all_labels).squeeze(1)

    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    targets = labels.numpy().astype(int)

    cm = confusion_matrix(targets, preds)
    row_sums = cm.sum(axis=1, keepdims=True)
    balanced_cm = cm / row_sums

    return np.diag(balanced_cm)

def evaluate(model, dataset, batch_size, device):
    mean_agg = _evaluate_one_agg(model, dataset, batch_size, Aggregation.MEAN, device)
    max_agg = _evaluate_one_agg(model, dataset, batch_size, Aggregation.MAX, device)
    return *mean_agg, *max_agg
    