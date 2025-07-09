from datasets_shim import *

def get_path(dataset):
    return f'/home/nikola.jovisic.ivi/nj/lustre_mock/{dataset}/embeddings.hdf5'

dataset_config = {
     get_path('embed'): {'pos': [5, 4], 'neg': [1]},
#    get_path('vindr'): {'pos': [5], 'neg': [1]},
#     get_path('csaw'): {'pos': [1], 'neg': [0]},
#     get_path('rsna'): {'pos': [2, 3], 'neg': [0]},
}

specificity, sensitivity = train_head(dataset_config)

print(specificity)
print(sensitivity)
