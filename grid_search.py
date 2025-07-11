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
#     'vindr': vindr,
#     'csaw': csaw,
#     'rsna': rsna,
#     'merged': merged
}

aggregations = [None, Aggregation.MAX, Aggregation.MEAN]
pos_weights = [1, 1.5]
batch_sizes = [128, 256, 512]
lrs = [5e-2, 1e-3, 5e-3]

def run_training(aggregation, pos_weight, gpu_id):
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

    for batch_size in batch_sizes:
        for lr in lrs:
            scaled_lr = lr * (batch_size // 128)
            for dataset_config_name in dataset_configs:
                cfg.train.aggregation = aggregation
                cfg.pos_weight = pos_weight
                cfg.train.fw_batch_size = batch_size
                cfg.train.bw_batch_size = batch_size
                cfg.train.lr = scaled_lr

                specificity_mean_agg, sensitivity_mean_agg, specificity_max_agg, sensitivity_max_agg = train_head(
                    dataset_configs[dataset_config_name],
                    cfg,
                    gpu_id
                )

                row_mean = [
                    aggregation.name if aggregation else None,
                    'MEAN',
                    pos_weight,
                    batch_size,
                    scaled_lr,
                    dataset_config_name,
                    specificity_mean_agg,
                    sensitivity_mean_agg
                ]
                
                row_max = [
                    aggregation.name if aggregation else None,
                    'MAX',
                    pos_weight,
                    batch_size,
                    scaled_lr,
                    dataset_config_name,
                    specificity_max_agg,
                    sensitivity_max_agg
                ]

                with open(output_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row_mean)
                    writer.writerow(row_max)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    agg_test_combos = list(itertools.product(aggregations, pos_weights))

    processes = []
    for i, (agg, test_agg) in enumerate(agg_test_combos):
        p = mp.Process(target=run_training, args=(agg, test_agg, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
