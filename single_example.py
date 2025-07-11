import csv
import itertools
import multiprocessing as mp
from datasets_shim import *

def get_path(dataset):
    return f'/home/nikola.jovisic.ivi/nj/lustre_mock/{dataset}/embeddings.hdf5'

embed = {
    get_path('embed'): {'pos': [5, 4], 'neg': [1]},
}

cfg = load_cfg()

cfg.train.aggregation = None
cfg.pos_weight = 1.5
cfg.train.fw_batch_size = 128
cfg.train.bw_batch_size = 128
cfg.train.lr = 0.005
#cfg.train.save_path = 'weights.pt'
cfg.train.load_path = 'weights.pt'

specificity_mean_agg, sensitivity_mean_agg, specificity_max_agg, sensitivity_max_agg = train_head(
    embed,
    cfg,
    just_evaluate=True
)


print(specificity_mean_agg, sensitivity_mean_agg, specificity_max_agg, sensitivity_max_agg)

cfg.train.aggregation = Aggregation('mean')
cfg.train.save_path = 'finetuned.pt'
cfg.train.load_path = 'weights.pt'

specificity_mean_agg, sensitivity_mean_agg, specificity_max_agg, sensitivity_max_agg = train_head(
    embed,
    cfg
)

print(specificity_mean_agg, sensitivity_mean_agg, specificity_max_agg, sensitivity_max_agg)
