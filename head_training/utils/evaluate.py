import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from aggregation import aggregate

def evaluate(model, collated_dataset, aggregation):
    if aggregation is None:
        raise ValueError("No sense in not aggregating when testing!")
    
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y, _, group in collated_dataset:
            x, group = x.to('cuda'), group.to('cuda')
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