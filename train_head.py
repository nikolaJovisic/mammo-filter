from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from torch_scatter import scatter_max, scatter_mean
from torch.utils.data import IterableDataset

class FlattenedGroupSubset(IterableDataset):
    def __init__(self, groupwise_dataset, subset_indices):
        self.dataset = groupwise_dataset
        self.subset_indices = subset_indices

    def __iter__(self):
        for group_idx in self.subset_indices:
            group, label, weight = self.dataset[group_idx]
            for i in range(group.size(0)):
                yield group[i].unsqueeze(0), label, weight


def evaluate_metrics(model, collated_dataset):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y, _, group in collated_dataset:
            x, group = x.to('cuda'), group.to('cuda')
            logits = model(x)

            logits = logits[group >= 0]
            group = group[group >= 0]

            max_logits = scatter_mean(logits, group, dim=0)

            all_logits.append(max_logits.cpu())
            all_labels.append(y)

    logits = torch.cat(all_logits).squeeze(1)
    labels = torch.cat(all_labels).squeeze(1)

    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    targets = labels.numpy().astype(int)

    cm = confusion_matrix(targets, preds)
    row_sums = cm.sum(axis=1, keepdims=True)
    balanced_cm = cm / row_sums

    return np.diag(balanced_cm), cm


def collate(dataset, fw_batch_size):
    iterator = iter(dataset)
    buffered_group = None  # to store lookahead group

    while True:
        x_list = []
        group_list = []
        y_list = []
        w_list = []

        instance_count = 0
        group_idx = 0

        while instance_count < fw_batch_size:
            if buffered_group is not None:
                x, y, w = buffered_group
                buffered_group = None
            else:
                try:
                    x, y, w = next(iterator)
                except StopIteration:
                    break  # no more data

            group_size = x.size(0)

            if instance_count + group_size > fw_batch_size:
                buffered_group = (x, y, w)  # 🔁 store for next round
                break

            x_list.append(x)
            group_ids = torch.full((group_size,), group_idx, dtype=torch.long)
            group_list.append(group_ids)
            y_list.append(y)
            w_list.append(w)

            instance_count += group_size
            group_idx += 1

        if instance_count == 0:
            break  # no more data to yield

        # Pad if needed
        if instance_count < fw_batch_size:
            pad_size = fw_batch_size - instance_count
            feat_dim = x_list[0].size(1) if x_list else 768
            x_pad = torch.zeros((pad_size, feat_dim))
            group_pad = torch.full((pad_size,), -1, dtype=torch.long)

            x_list.append(x_pad)
            group_list.append(group_pad)

        x_batch = torch.cat(x_list, dim=0)  # (fw_batch_size, 768)
        group = torch.cat(group_list, dim=0)  # (fw_batch_size,)
        y = torch.stack(y_list) if y_list else torch.empty((0,))
        w = torch.stack(w_list) if w_list else torch.empty((0,))

        yield x_batch, y, w, group


def linear_probe(train_dataset, valid_dataset, test_dataset, batch_size, lr, epochs, patience):
    fw_batch_size = batch_size
    bw_batch_size = batch_size

    model = nn.Linear(768, 1).to('cuda')
    model.load_state_dict(torch.load('flattened_pretrained.pt'))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_val_loss = float('inf')
    patience_counter = 0

    y_pred_batch = []
    y_true_batch = []
    w_batch = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y, w, group in collate(train_dataset, fw_batch_size):
            x, y, w, group = x.to('cuda'), y.to('cuda'), w.to('cuda'), group.to('cuda')

            logits = model(x)
            
            logits = logits[group >= 0]
            group = group[group >= 0]

            max_logits = scatter_mean(logits, group, dim=0)

            y_pred_batch.append(max_logits)
            y_true_batch.append(y)
            w_batch.append(w)

            if sum(t.size(0) for t in y_pred_batch) >= bw_batch_size:
                group_sizes = [t.size(0) for t in y_pred_batch]
                cumulative = np.cumsum(group_sizes)

                # Find how many groups we can fit
                num_to_use = np.searchsorted(cumulative, bw_batch_size, side='right')

                # Get full groups
                used_preds = torch.cat(y_pred_batch[:num_to_use], dim=0)
                used_trues = torch.cat(y_true_batch[:num_to_use], dim=0)
                used_weights = torch.cat(w_batch[:num_to_use], dim=0)

                # Pad if needed
                pad_size = bw_batch_size - used_preds.size(0)
                if pad_size > 0:
                    device = used_preds.device
                    pad_pred = torch.zeros((pad_size, 1), device=device, requires_grad=True)
                    pad_true = torch.zeros((pad_size, 1), device=device)
                    pad_weight = torch.zeros((pad_size, 1), device=device)  # will zero out loss

                    used_preds = torch.cat([used_preds, pad_pred], dim=0)
                    used_trues = torch.cat([used_trues, pad_true], dim=0)
                    used_weights = torch.cat([used_weights, pad_weight], dim=0)

                # Compute loss
                loss = criterion(used_preds, used_trues)
                loss = (loss * used_weights).mean()

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                total_loss += loss.item()

                # Retain remaining groups
                y_pred_batch = y_pred_batch[num_to_use:]
                y_true_batch = y_true_batch[num_to_use:]
                w_batch = w_batch[num_to_use:]

        y_pred_batch = []
        y_true_batch = []
        w_batch = []

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, w, group in collate(valid_dataset, fw_batch_size):
                x, y, w, group = x.to('cuda'), y.to('cuda'), w.to('cuda'), group.to('cuda')

                logits = model(x)
                
                logits = logits[group >= 0]
                group = group[group >= 0]

                max_logits = scatter_mean(logits, group, dim=0)

                y_pred_batch.append(max_logits)
                y_true_batch.append(y)
                w_batch.append(w)

                if sum(t.size(0) for t in y_pred_batch) >= bw_batch_size:
                    group_sizes = [t.size(0) for t in y_pred_batch]
                    cumulative = np.cumsum(group_sizes)

                    # Find how many groups we can fit
                    num_to_use = np.searchsorted(cumulative, bw_batch_size, side='right')

                    # Get full groups
                    used_preds = torch.cat(y_pred_batch[:num_to_use], dim=0)
                    used_trues = torch.cat(y_true_batch[:num_to_use], dim=0)
                    used_weights = torch.cat(w_batch[:num_to_use], dim=0)

                    # Pad if needed
                    pad_size = bw_batch_size - used_preds.size(0)
                    if pad_size > 0:
                        device = used_preds.device
                        pad_pred = torch.zeros((pad_size, 1), device=device, requires_grad=True)
                        pad_true = torch.zeros((pad_size, 1), device=device)
                        pad_weight = torch.zeros((pad_size, 1), device=device)  # will zero out loss

                        used_preds = torch.cat([used_preds, pad_pred], dim=0)
                        used_trues = torch.cat([used_trues, pad_true], dim=0)
                        used_weights = torch.cat([used_weights, pad_weight], dim=0)

                    # Compute loss
                    loss = criterion(used_preds, used_trues)
                    loss = (loss * used_weights).mean()

                    val_loss += loss.item()

                    # Retain remaining groups
                    y_pred_batch = y_pred_batch[num_to_use:]
                    y_true_batch = y_true_batch[num_to_use:]
                    w_batch = w_batch[num_to_use:]

        if val_loss < best_val_loss:
            best_val_loss = val_loss 
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    #torch.save(best_model_state, 'embed_flattened_pretrained.pt')
    model.load_state_dict(best_model_state)
    return evaluate_metrics(model, collate(test_dataset, fw_batch_size))

def train_head(dataset, n_runs=1, batch_size=128, lr=1e-3, epochs=100, patience=5):
    labels = np.array([dataset[i][1].item() for i in range(len(dataset))])

    train_idx, temp_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.3,
        stratify=labels,
        random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=labels[temp_idx],
        random_state=42
    )

    splits = {'Train': train_idx, 'Validation': val_idx, 'Test': test_idx}
    for name, idx in splits.items():
        y = labels[idx]
        pos, neg = (y == 1).sum(), (y == 0).sum()
        print(f'{name} split: {len(y)} samples (Pos: {pos}, Neg: {neg})')

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    

    diags = []
    for i in range(n_runs):
        print(f'\n=== Run {i+1}/{n_runs} ===')
        diag, cm = linear_probe(train_dataset, valid_dataset, test_dataset, batch_size, lr, epochs, patience)
        print(cm)
        diags.append(diag)

    diags = np.stack(diags)
    mean_diag = diags.mean(axis=0)
    std_diag = diags.std(axis=0)

    for i, cls in enumerate(['negatives labeled as negative', 'positives labeled as positive']):
        mean_val = mean_diag[i]
        std_val = std_diag[i]
        if std_val / mean_val > 0.05:
            print(f'{mean_val:.4f} ± {std_val:.4f} of {cls}.')
        else:
            print(f'{mean_val:.4f} of {cls}.')
                                                      
