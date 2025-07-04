import torch
import torch.nn as nn
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class EmbeddingsDataset(Dataset):
    def __init__(self, path, pos_labels, neg_labels, pos_weight=1.0):
        self.embeddings = []
        self.labels = []
        self.class_counts = defaultdict(int)

        with h5py.File(path, 'r') as f:
            for group_name in f:
                group = f[group_name]
                embs = group['images'][()]
                raw_label = int(group['label'][()])

                if raw_label in pos_labels:
                    label = 0
                elif raw_label in neg_labels:
                    label = 1
                else:
                    continue
                    
                self.embeddings.append(embs)
                self.labels.extend([label] * embs.shape[0])
                self.class_counts[label] += embs.shape[0]
                
        embs_np = np.concatenate(self.embeddings)
        embs_tensor = torch.tensor(embs_np, dtype=torch.float32)
        self.embeddings = torch.nn.functional.normalize(embs_tensor, p=2, dim=1) # L2 normalization

        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)
        self.sample_weights = self._compute_sample_weights(pos_weight)
        
        print(len(self.labels))

    def _compute_sample_weights(self, positive_class_weight_multiplier=1.0):
        total = sum(self.class_counts.values())
        weights = {
            cls: (total / (2 * count)) * (positive_class_weight_multiplier if cls == 1 else 1)
            for cls, count in self.class_counts.items()
        }
        weight_list = [weights[int(lbl.item())] for lbl in self.labels]
        weights_tensor = torch.tensor(weight_list, dtype=torch.float32).unsqueeze(1)
        return torch.clamp(weights_tensor, min=1e-6)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.sample_weights[idx]