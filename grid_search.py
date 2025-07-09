import csv
import itertools
import multiprocessing as mp
from datasets_shim import *

def get_path(dataset):
    return f'/home/nikola.jovisic.ivi/nj/lustre_mock/{dataset}/embeddings.hdf5'

embed = {
    get_path('embed'): {'pos': [5, 4], 'neg': [1]},
}
vindr = {
    get_path('vindr'): {'pos': [5, 4], 'neg': [1]},
}
csaw = {
    get_path('csaw'): {'pos': [1], 'neg': [0]},
}
rsna = {
    get_path('rsna'): {'pos': [3], 'neg': [0]},
}
merged = {
    get_path('embed'): {'pos': [5, 4], 'neg': [1]},
    get_path('vindr'): {'pos': [5, 4], 'neg': [1]},
    get_path('csaw'): {'pos': [1], 'neg': [0]},
    get_path('rsna'): {'pos': [3], 'neg': [0]},
}

dataset_configs = {
    'embed': embed,
    'vindr': vindr,
    'csaw': csaw,
    'rsna': rsna,
    'merged': merged
}

aggregations = [None, Aggregation.MAX, Aggregation.MEAN]
test_aggregations = [Aggregation.MAX, Aggregation.MEAN]
pos_weights = [1, 1.5, 2]
batch_sizes = [128, 256, 512]
lrs = [5e-4, 1e-3, 5e-3]

def run_training(aggregation, test_aggregation, gpu_id):
    import torch
    torch.cuda.set_device(gpu_id)

    cfg = load_cfg()
    output_file = f"results_gpu{gpu_id}.csv"

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'aggregation', 'test_aggregation', 'pos_weight', 'batch_size', 'lr',
            'dataset_config_name', 'specificity', 'sensitivity'
        ])

    for pos_weight in pos_weights:
        for batch_size in batch_sizes:
            for lr in lrs:
                scaled_lr = lr * (batch_size // 128)
                for dataset_config_name in dataset_configs:
                    cfg.aggregation = aggregation
                    cfg.test_aggregation = test_aggregation
                    cfg.pos_weight = pos_weight
                    cfg.fw_batch_size = batch_size
                    cfg.bw_batch_size = batch_size
                    cfg.lr = scaled_lr

                    specificity, sensitivity = train_head(
                        dataset_configs[dataset_config_name],
                        cfg,
                        gpu_id
                    )

                    row = [
                        aggregation.name if aggregation else None,
                        test_aggregation.name,
                        pos_weight,
                        batch_size,
                        scaled_lr,
                        dataset_config_name,
                        specificity,
                        sensitivity
                    ]

                    with open(output_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(row)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    agg_test_combos = list(itertools.product(aggregations, test_aggregations))

    processes = []
    for i, (agg, test_agg) in enumerate(agg_test_combos):
        p = mp.Process(target=run_training, args=(agg, test_agg, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
