import torch

def collate(dataset, batch_size):
    iterator = iter(dataset)
    buffered_group = None  # to store lookahead group

    while True:
        x_list = []
        group_list = []
        y_list = []
        w_list = []

        instance_count = 0
        group_idx = 0

        while instance_count < batch_size:
            if buffered_group is not None:
                x, y, w = buffered_group
                buffered_group = None
            else:
                try:
                    x, y, w = next(iterator)
                except StopIteration:
                    break  # no more data

            group_size = x.size(0)

            if instance_count + group_size > batch_size:
                buffered_group = (x, y, w)  # store for next round
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
        if instance_count < batch_size:
            pad_size = batch_size - instance_count
            feat_dim = x_list[0].size(1) if x_list else 768
            x_pad = torch.zeros((pad_size, feat_dim))
            group_pad = torch.full((pad_size,), -1, dtype=torch.long)

            x_list.append(x_pad)
            group_list.append(group_pad)

        x_batch = torch.cat(x_list, dim=0)  # (batch_size, 768)
        group = torch.cat(group_list, dim=0)  # (batch_size,)
        y = torch.stack(y_list) if y_list else torch.empty((0,))
        w = torch.stack(w_list) if w_list else torch.empty((0,))

        yield x_batch, y, w, group