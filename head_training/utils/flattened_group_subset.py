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