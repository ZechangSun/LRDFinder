import os
import numpy as np
from typing import Optional, Tuple, TypeAlias, List
from datetime import datetime



import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torch.nn.functional as F


import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from astropy.io import fits

from .model import LowRankCNN, LowRankMLP, OHEMBCEWithLogitsLoss, BPRLoss
from .dataset import LRDDataset


import swanlab
from swanlab.integration.pytorch_lightning import SwanLabLogger




class LRDFinder(pl.LightningModule):
    def __init__(self, 
                 learning_rate=1e-4,
                 npixels=255,
                 random_mask_ratio=0.1,
                 random_shift=(-25, 75),
                 noise_std_max=0.05,
                 model_type='lowrank_cnn',
                 hidden_dims=None,
                 rank=16,
                 dropout_rate=0.2,
                 task='classification'):
        super(LRDClassifier, self).__init__()
        self.save_hyperparameters()
        self._initialize_model()
        self._initialize_loss()
        self.training_step_outputs, self.validation_step_outputs = [], []
    
    def forward(self, x):
        if isinstance(x, dict):
            spec, error = x['spec'], x['error']
            if self.hparams.model_type == 'lowrank_cnn':
                input_x = torch.stack([spec, error], dim=1)
            elif self.hparams.model_type == 'lowrank_mlp':
                input_x = spec
            else:
                raise ValueError(f"Unknown model type: {self.hparams.model_type}")
        elif isinstance(x, torch.Tensor):
            input_x = x
        else:
            raise ValueError("Input x must be a dict or torch.Tensor")
        return self.model(input_x)
    
    def training_step(self, batch, batch_idx):
        if self.hparams.task == 'classification':
            return self._classification_training_step(batch, batch_idx)
        elif self.hparams.task == 'rank':
            return self._rank_training_step(batch, batch_idx)
        else:
            raise ValueError(f"Unknown task type: {self.hparams.task}")
    
    def validation_step(self, batch, batch_idx):
        y = batch['label'].float()
        y_hat = self(batch).squeeze()

        self.validation_step_outputs.append({
            'preds': y_hat.detach(),
            'targets': y.detach(),
        })

        return loss
    
    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
            all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])

            positive_mask = (all_targets == 1)
            sorted_indices = torch.argsort(all_preds, descending=True)
            rank = torch.zeros_like(all_preds, dtype=torch.int32)
            rank[sorted_indices] = torch.arange(1, len(all_preds) + 1, device=all_preds.device)
            positive_rank = rank[positive_mask]

            swanlab.log({
                'max_positive_rank': positive_rank.max().item(),
                'mean_positive_rank': positive_rank.float().mean().item(),
                'min_positive_rank': positive_rank.min().item(),
            })
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def _initialize_model(self):
        if self.hparams.model_type == 'lowrank_cnn':
            self.model = LowRankCNN(
                input_length=self.hparams.npixels,
                conv_channels=[16, 32, 64],
                fc_dims=self.hparams.hidden_dims if self.hparams.hidden_dims is not None else [128, 64],
                rank=self.hparams.rank,
                num_classes=1,
                dropout_rate=self.hparams.dropout_rate,
                activation='relu'
            )
        elif self.model_type == 'lowrank_mlp':
            self.model = LowRankMLP(
                input_dim=self.hparams.npixels,
                hidden_dims=self.hparams.hidden_dims if self.hparams.hidden_dims is not None else [256, 128, 64],
                rank=self.hparams.rank,
                num_classes=1,  
                dropout_rate=self.hparams.dropout_rate
            )
        else:
            raise ValueError(f"Unknown model type: {self.hparams.model_type}")
    
    def _initialize_loss(self):
        if self.hparams.task == 'classification':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.hparams.task == 'rank':
            self.loss_fn = BPRLoss()
        else:
            raise ValueError(f"Unknown task type: {self.hparams.task}")
    

    def _classification_training_step(self, batch, batch_idx, stage='train'):
        y = batch['label'].float()

        y_hat = self(batch).squeeze()
        loss = self.loss_fn(y_hat, y)
        
        if stage == 'train':
            swanlab.log({
                "train_loss": loss.item(),
                "step": self.global_step
            })
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return loss
    
    def _rank_training_step(self, batch, batch_idx):
        pos_batch = {
            'spec': batch['pos_spec'],
            'error': batch['pos_error'],
            'label': batch['pos_label'],
        }
        neg_batch = {
            'spec': batch['neg_spec'],
            'error': batch['neg_error'],
            'label': batch['neg_label'],
        }
        pos_scores = self(pos_batch).squeeze()
        neg_scores = self(neg_batch).squeeze()
        
        loss = self.loss_fn(pos_score, neg_score)
        if stage == 'train':
            swanlab.log({
                "train_loss": loss.item(),
                "step": self.global_step
            })
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return loss
