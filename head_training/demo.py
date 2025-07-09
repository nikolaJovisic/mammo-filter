from datasets_shim import *
from batched_dataloader import get_batched_dataloader, BatchEnum
from embedding_inference import EmbeddingInference
from embeddings_dataset import EmbeddingsDataset
from train_head import train_head
from itertools import islice
from torch.utils.data import DataLoader

def get_path(dataset):
    return f'/home/nikola.jovisic.ivi/nj/lustre_mock/{dataset}/embeddings.hdf5'

dataset_config = {
   get_path('vindr'): {'pos': [5], 'neg': [1]},
#      get_path('embed'): {'pos': [5], 'neg': [1]},
#     get_path('csaw'): {'pos': [1], 'neg': [0]},
#     get_path('rsna'): {'pos': [2, 3], 'neg': [0]},
}

#ovo sa birads 1 i birads 5 je vrv dobro za dokazivanje MIL-a po dojci

dataset = EmbeddingsDataset(dataset_config, groupwise=False)

train_head(dataset, n_runs=1)
