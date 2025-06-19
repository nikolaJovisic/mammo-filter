import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import h5py
from pathlib import Path
from model.build import build_model
from utils.serialization import save_embedding_inference

class EmbeddingInference:
    def __init__(self, dataset, cfg_path=None):        
        
        if cfg_path is None:
            cfg_path = Path(__file__).parent / "config.yaml"

        self.cfg = OmegaConf.load(cfg_path)
        
        if self.cfg.model.img_size != dataset.
        
        self.model = self._build_model()
        self.dataset = dataset
        self.loader = DataLoader(self.dataset, batch_size=cfg.batch_size)
        self.output_dir = os.path.join(cfg.embeddings_path, cfg.run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.hdf5_out_path = os.path.join(self.output_dir, 'embeddings.hdf5')
        save_embedding_inference(self, os.path.join(self.output_dir, 'config.yaml'))

    def _build_model(self):
        model = build_model(self.cfg.model)
        state_dict = torch.load(self.cfg.model.weights, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval().to('cuda')
        return model

    def run(self):
        with h5py.File(self.hdf5_out_path, 'w') as h5f:
            embeddings_ds = None
            labels_ds = None
            idx = 0
            with torch.no_grad():
                for images, labels in tqdm(self.loader):
                    images = images.to('cuda')
                    embeddings = self.model(images).cpu()
                    batch_size = embeddings.shape[0]
                    if embeddings_ds is None:
                        embedding_dim = embeddings.shape[1]
                        embeddings_ds = h5f.create_dataset('embeddings', shape=(len(self.dataset), embedding_dim), dtype='f4')
                        labels_ds = h5f.create_dataset('labels', shape=(len(self.dataset),), dtype='i8')
                    embeddings_ds[idx:idx + batch_size] = embeddings.numpy()
                    labels_ds[idx:idx + batch_size] = labels.numpy()
                    idx += batch_size
