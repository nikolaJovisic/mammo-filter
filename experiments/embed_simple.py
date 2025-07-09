import sys 

sys.path.append('..')

from datasets_shim import *
from batched_dataloader import get_batched_dataloader, BatchEnum
from embedding_inference import EmbeddingInference
from itertools import islice
from icecream import ic
    
args = {'return_mode': ReturnMode.BREAST_TILES_LABEL, 'convert_to': ConvertTo.RGB_TENSOR_IMGNET_NORM, 'tile_size': 518, 'resize': 518, 'final_resize': 518}

# EmbeddingInference(MammoDataset(DatasetEnum.CSAW, labels=[0, 1, 2], **args), run_id='csaw').run_images()
# EmbeddingInference(MammoDataset(DatasetEnum.RSNA, labels=[0, 1, 2, 3], read_window=True, **args), run_id='rsna').run_images()
EmbeddingInference(MammoDataset(DatasetEnum.VINDR, read_window=True, **args), run_id='vindr').run_images()
EmbeddingInference(MammoDataset(DatasetEnum.EMBED, labels=[1, 2, 3, 4, 5], **args), run_id='embed').run_images()