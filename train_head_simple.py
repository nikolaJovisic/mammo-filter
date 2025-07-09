from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.metrics import confusion_matrix

def evaluate_metrics(model, loader, split):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to('cuda'), y.to('cuda')
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    logits = torch.cat(all_logits).squeeze(1)
    labels = torch.cat(all_labels).squeeze(1)

    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    targets = labels.numpy().astype(int)

    cm = confusion_matrix(targets, preds)
    row_sums = cm.sum(axis=1, keepdims=True)
    balanced_cm = cm / row_sums

    return np.diag(balanced_cm)


def linear_probe(train_dataset, valid_dataset, test_dataset, batch_size, lr, epochs, patience):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = nn.Linear(768, 1).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y, w in train_loader:
            x, y, w = x.to('cuda'), y.to('cuda'), w.to('cuda')
            logits = model(x)
            loss = criterion(logits, y)
            loss = (loss * w).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, w in valid_loader:
                x, y, w = x.to('cuda'), y.to('cuda'), w.to('cuda')
                logits = model(x)
                loss = criterion(logits, y)
                loss = (loss * w).mean()
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model_state)
    return evaluate_metrics(model, test_loader, 'Test')


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
        diag = linear_probe(train_dataset, valid_dataset, test_dataset, batch_size, lr, epochs, patience)
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