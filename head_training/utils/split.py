import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from utils.flattened_group_subset import FlattenedGroupSubset

def split(dataset, valid_size, test_size, flatten):
    labels = np.array([dataset[i][1].item() for i in range(len(dataset))])

    train_idx, temp_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=(valid_size + test_size),
        stratify=labels,
        random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(test_size / (valid_size + test_size)),
        stratify=labels[temp_idx],
        random_state=42
    )

    splits = {'Train': train_idx, 'Validation': val_idx, 'Test': test_idx}
    for name, idx in splits.items():
        y = labels[idx]
        pos, neg = (y == 1).sum(), (y == 0).sum()
        print(f'{name} split: {len(y)} samples (Pos: {pos}, Neg: {neg})')

    if flatten:
        train_ds = FlattenedGroupSubset(dataset, train_idx)
        valid_ds = FlattenedGroupSubset(dataset, val_idx)
    else:
        train_ds = Subset(dataset, train_idx)
        valid_ds = Subset(dataset, val_idx)
        
    test_ds = Subset(dataset, test_idx)
    
    return train_ds, valid_ds, test_ds