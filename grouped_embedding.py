class GroupedEmbeddingDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        out = self.base[idx]

        if isinstance(out, tuple) or isinstance(out, list):
            images = extract_images(out, self.base.return_mode)
            label = extract_label(out, self.base.return_mode)
        else:
            images = [out]
            label = None

        return images, idx, label

def embedding_collate(batch):
    batch_images = []
    batch_group_idx = []
    batch_labels = []

    for item in batch:
        images, group_idx, label = item
        batch_images.append(images)
        batch_group_idx.append(group_idx)
        batch_labels.append(label)

    return batch_images, batch_group_idx, batch_labels
