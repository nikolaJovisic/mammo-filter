from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from omegaconf import OmegaConf
from embeddings_dataset import EmbeddingsDataset
from utils.split_collate import split
from utils.aggregation import aggregate
        
def _train(train_dataset, valid_dataset, cfg):
    model = nn.Linear(768, 1).to('cuda')
    
    if cfg.load_path is not None:
        model.load_state_dict(torch.load(cfg.load_path))
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_val_loss = float('inf')
    patience_counter = 0

    y_pred_batch = []
    y_true_batch = []
    w_batch = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y, w, group in train_dataset:
            x, y, w, group = x.to('cuda'), y.to('cuda'), w.to('cuda'), group.to('cuda')

            logits = model(x)
            
            logits = logits[group >= 0]
            group = group[group >= 0]

            aggregated_logits = aggregate(logits, group, cfg.aggregation)

            y_pred_batch.append(aggregated_logits)
            y_true_batch.append(y)
            w_batch.append(w)

            if sum(t.size(0) for t in y_pred_batch) >= cfg.bw_batch_size:
                group_sizes = [t.size(0) for t in y_pred_batch]
                cumulative = np.cumsum(group_sizes)

                # Find how many groups we can fit
                num_to_use = np.searchsorted(cumulative, cfg.bw_batch_size, side='right')

                # Get full groups
                used_preds = torch.cat(y_pred_batch[:num_to_use], dim=0)
                used_trues = torch.cat(y_true_batch[:num_to_use], dim=0)
                used_weights = torch.cat(w_batch[:num_to_use], dim=0)

                # Pad if needed
                pad_size = cfg.bw_batch_size - used_preds.size(0)
                if pad_size > 0:
                    device = used_preds.device
                    pad_pred = torch.zeros((pad_size, 1), device=device, requires_grad=True)
                    pad_true = torch.zeros((pad_size, 1), device=device)
                    pad_weight = torch.zeros((pad_size, 1), device=device)  # will zero out loss

                    used_preds = torch.cat([used_preds, pad_pred], dim=0)
                    used_trues = torch.cat([used_trues, pad_true], dim=0)
                    used_weights = torch.cat([used_weights, pad_weight], dim=0)

                # Compute loss
                loss = criterion(used_preds, used_trues)
                loss = (loss * used_weights).mean()

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                total_loss += loss.item()

                # Retain remaining groups
                y_pred_batch = y_pred_batch[num_to_use:]
                y_true_batch = y_true_batch[num_to_use:]
                w_batch = w_batch[num_to_use:]

        y_pred_batch = []
        y_true_batch = []
        w_batch = []

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, w, group in valid_dataset:
                x, y, w, group = x.to('cuda'), y.to('cuda'), w.to('cuda'), group.to('cuda')

                logits = model(x)
                
                logits = logits[group >= 0]
                group = group[group >= 0]

                aggregated_logits = aggregate(logits, group, cfg.aggregation)

                y_pred_batch.append(aggregated_logits)
                y_true_batch.append(y)
                w_batch.append(w)

                if sum(t.size(0) for t in y_pred_batch) >= cfg.bw_batch_size:
                    group_sizes = [t.size(0) for t in y_pred_batch]
                    cumulative = np.cumsum(group_sizes)

                    # Find how many groups we can fit
                    num_to_use = np.searchsorted(cumulative, cfg.bw_batch_size, side='right')

                    # Get full groups
                    used_preds = torch.cat(y_pred_batch[:num_to_use], dim=0)
                    used_trues = torch.cat(y_true_batch[:num_to_use], dim=0)
                    used_weights = torch.cat(w_batch[:num_to_use], dim=0)

                    # Pad if needed
                    pad_size = cfg.bw_batch_size - used_preds.size(0)
                    if pad_size > 0:
                        device = used_preds.device
                        pad_pred = torch.zeros((pad_size, 1), device=device, requires_grad=True)
                        pad_true = torch.zeros((pad_size, 1), device=device)
                        pad_weight = torch.zeros((pad_size, 1), device=device)  # will zero out loss

                        used_preds = torch.cat([used_preds, pad_pred], dim=0)
                        used_trues = torch.cat([used_trues, pad_true], dim=0)
                        used_weights = torch.cat([used_weights, pad_weight], dim=0)

                    # Compute loss
                    loss = criterion(used_preds, used_trues)
                    loss = (loss * used_weights).mean()

                    val_loss += loss.item()

                    # Retain remaining groups
                    y_pred_batch = y_pred_batch[num_to_use:]
                    y_true_batch = y_true_batch[num_to_use:]
                    w_batch = w_batch[num_to_use:]

        if val_loss < best_val_loss:
            best_val_loss = val_loss 
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if cfg.save_path is not None:
        torch.save(best_model_state, cfg.save_path)
        
    model.load_state_dict(best_model_state)
    return model

def _load_cfg(cfg_path):
    if cfg_path is None:
        cfg_path = Path(__file__).parent / "config.yaml"
    return OmegaConf.load(cfg_path)

def train_head(dataset_config, cfg_path=None)
    cfg = _load_cfg(cfg_path)
    dataset = EmbeddingDataset(dataset_config, cfg.pos_weight)
    
    train_ds, valid_ds, test_ds = split_collate(dataset, cfg.valid_size, cfg.test_size, 
                                                cfg.train.aggregation is None, cfg.fw_batch_size)
    
    model = _train(train_ds, valid_ds, cfg.train)
    specificity, sensitivity = evaluate(model, test_ds, cfg.test_aggregation)