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

from LRDFinder.model import LowRankCNN, LowRankMLP, OHEMBCEWithLogitsLoss
from LRDFinder.dataset import LRDDataset


SWANLAB_ROOT = "/home/zechang/user/DESI_LRD_ML/swanlab_data"
os.environ["SWANLAB_SAVE_DIR"] = SWANLAB_ROOT 
os.makedirs(SWANLAB_ROOT, exist_ok=True) 

import swanlab
from swanlab.integration.pytorch_lightning import SwanLabLogger




class LRDClassifier(pl.LightningModule):
    def __init__(self, 
                 learning_rate=1e-4,
                 npixels=255,
                 random_mask_ratio=0.1,
                 random_shift=(-25, 75),
                 noise_std_max=0.05,
                 model_type='lowrank_cnn',
                 hidden_dims=None,
                 rank=16,
                 dropout_rate=0.2):
        super(LRDClassifier, self).__init__()
        self.save_hyperparameters()

        if model_type == 'lowrank_cnn':
            self.model = LowRankCNN(
                input_length=npixels,
                conv_channels=[16, 32, 64],
                fc_dims=hidden_dims if hidden_dims is not None else [128, 64],
                rank=rank,
                num_classes=1,
                dropout_rate=dropout_rate,
                activation='relu'
            )
        elif model_type == 'lowrank_mlp':
            self.model = LowRankMLP(
                input_dim=npixels,
                hidden_dims=hidden_dims if hidden_dims is not None else [256, 128, 64],
                rank=rank,
                num_classes=1,
                dropout_rate=dropout_rate
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
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
        y = batch['label'].float()

        y_hat = self(batch).squeeze()
        loss = self.loss_fn(y_hat, y)
        
        pred = (torch.sigmoid(y_hat) > 0.5).float()
        acc = (pred == y).float().mean()
        
        self.training_step_outputs.append({
            'preds': torch.sigmoid(y_hat).detach(),
            'targets': y.detach()
        })
        
        swanlab.log({
            "train_loss": loss.item(),
            "train_acc": acc.item(),
            "step": self.global_step
        })
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return loss
    
    def on_train_epoch_end(self):
        if self.training_step_outputs:
            all_preds = torch.cat([x['preds'] for x in self.training_step_outputs])
            all_targets = torch.cat([x['targets'] for x in self.training_step_outputs])
            
            preds_np = all_preds.cpu().numpy()
            targets_np = all_targets.cpu().numpy()
            preds_binary = (preds_np > 0.5).astype(int)

            try:
                f1 = f1_score(targets_np, preds_binary, zero_division=0)
                recall = recall_score(targets_np, preds_binary, zero_division=0)
                precision = precision_score(targets_np, preds_binary, zero_division=0)
                auc = roc_auc_score(targets_np, preds_np) if len(np.unique(targets_np)) > 1 else 0.0
            except:
                f1 = recall = precision = auc = 0.0
            
            swanlab.log({
                "train_f1": f1,
                "train_recall": recall,
                "train_precision": precision,
                "train_auc": auc,
                "epoch": self.current_epoch
            })
            
            self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        y = batch['label'].float()
        y_hat = self(batch).squeeze()
        loss = self.loss_fn(y_hat, y)
        
        pred = (torch.sigmoid(y_hat) > 0.5).float()
        acc = (pred == y).float().mean()
        
        self.validation_step_outputs.append({
            'preds': torch.sigmoid(y_hat).detach(),
            'targets': y.detach(),
            'val_loss': loss.detach(),
            'val_acc': acc.detach()
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
            all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
            avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
            avg_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean()
            
            preds_np = all_preds.cpu().numpy()
            targets_np = all_targets.cpu().numpy()
            preds_binary = (preds_np > 0.5).astype(int)

            try:
                f1 = f1_score(targets_np, preds_binary, zero_division=0)
                recall = recall_score(targets_np, preds_binary, zero_division=0)
                precision = precision_score(targets_np, preds_binary, zero_division=0)
                auc = roc_auc_score(targets_np, preds_np) if len(np.unique(targets_np)) > 1 else 0.0
            except:
                f1 = recall = precision = auc = 0.0

            swanlab.log({
                "val_loss": avg_loss.item(),
                "val_acc": avg_acc.item(),
                "val_f1": f1,
                "val_recall": recall,
                "val_precision": precision,
                "val_auc": auc,
                "epoch": self.current_epoch
            })
            
            self.log('val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_acc', avg_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_recall', recall, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_precision', precision, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_auc', auc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            
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



def main():
    POS_FITS_FILES = [
        '/home/zechang/user/DESI_LRD_ML/data/mock_spectra_Ha_with_abs.fits'
    ]
    NEG_FITS_FILES = [
        #'/home/zechang/user/DESI_LRD_ML/data/mock_spectra_Hb_narrow_only.fits'
        #'/home/zechang/user/DESI_LRD_ML/data/mock_spectra_Hb_no_abs.fits'
        '/home/zechang/user/DESI_LRD_ML/data/real_spectra_Hb_psf_100000.fits'
    ]

    BATCH_SIZE = 32
    MAX_EPOCHS = 50
    LEARNING_RATE = 1e-5
    NPIXELS = 255
    RANDOM_MASK_RATIO = 0.1
    RANDOM_SHIFT = (-25, 75)
    NOISE_STD_MAX = 0.05
    
    MODEL_TYPE = 'lowrank_cnn'
    HIDDEN_DIMS = [256, 128, 64]
    RANK = 128
    DROPOUT_RATE = 0.2
    
    SWAN_PROJECT = "lrd_classification"
    SWAN_SAVE_DIR = "/home/zechang/user/DESI_LRD_ML/swanlab_logs"
    EXPERIMENT_NAME = f"lrd_{MODEL_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    swanlogger = SwanLabLogger(
        project=SWAN_PROJECT,
        experiment_name=EXPERIMENT_NAME,
        save_dir=SWAN_SAVE_DIR,
        api_key="kSQ0WETobYCojrxzRkZCX"
    )

    swanlab.config.update({
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "npixels": NPIXELS,
        "random_mask_ratio": RANDOM_MASK_RATIO,
        "random_shift": RANDOM_SHIFT,
        "noise_std_max": NOISE_STD_MAX,
        "model_type": MODEL_TYPE,
        "hidden_dims": HIDDEN_DIMS,
        "rank": RANK,
        "dropout_rate": DROPOUT_RATE,
        "sampling_strategy": "weighted_random_sampler"
    })
    

    full_dataset = LRDDataset(
        pos_fits_files=POS_FITS_FILES,
        neg_fits_files=NEG_FITS_FILES,
        npixels=NPIXELS,
        random_mask_ratio=RANDOM_MASK_RATIO,
        random_shift=RANDOM_SHIFT,
        noise_std_max=NOISE_STD_MAX,
        is_train=True
    )
    

    full_dataset = LRDDataset(
    pos_fits_files=POS_FITS_FILES,
    neg_fits_files=NEG_FITS_FILES,
    npixels=NPIXELS,
    random_mask_ratio=RANDOM_MASK_RATIO,
    random_shift=RANDOM_SHIFT,
    noise_std_max=NOISE_STD_MAX,
    is_train=True
)

    # ===================== 核心修改：手动拆分正负样本，保证验证集1:1 =====================
    # 1. 分离正/负样本的索引
    pos_idx = np.where(full_dataset.labels == 1)[0]  # 所有正样本索引
    neg_idx = np.where(full_dataset.labels == 0)[0]  # 所有负样本索引

    # 2. 设置随机种子，保证结果可复现
    np.random.seed(42)

    # 3. 定义验证集总目标大小（保持原test_size=0.2的比例）
    total_samples = len(full_dataset)
    val_target_size = 2000
    val_per_class = val_target_size // 2        # 验证集每类（正/负）的目标数量

    # 4. 确保每类样本数不超过实际存在的数量（边界保护）
    val_pos_num = min(val_per_class, len(pos_idx))  # 验证集正样本数（不超过总正样本）
    val_neg_num = min(val_per_class, len(neg_idx))  # 验证集负样本数（不超过总负样本）

    # 5. 随机打乱正负样本索引（保证随机性）
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    # 6. 拆分验证集/训练集的正负样本索引
    # 验证集：取前val_pos_num个正样本 + 前val_neg_num个负样本
    val_pos_idx = pos_idx[:val_pos_num]
    val_neg_idx = neg_idx[:val_neg_num]
    val_indices = np.concatenate([val_pos_idx, val_neg_idx])

    # 训练集：取剩余的正样本 + 剩余的负样本
    train_pos_idx = pos_idx[val_pos_num:]
    train_neg_idx = neg_idx[val_neg_num:]
    train_indices = np.concatenate([train_pos_idx, train_neg_idx])

    # ===================== 后续代码保持不变 =====================
    # 创建子集
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # 统计并打印信息
    train_labels = full_dataset.labels[train_indices]
    pos_count = np.sum(train_labels == 1)
    neg_count = len(train_labels) - pos_count

    val_labels = full_dataset.labels[val_indices]
    val_pos_count = np.sum(val_labels == 1)
    val_neg_count = len(val_labels) - val_pos_count

    print(f"===== 数据集统计 =====")
    print(f"总样本数: {len(full_dataset)} (正: {np.sum(full_dataset.labels==1)}, 负: {np.sum(full_dataset.labels==0)})")
    print(f"训练集样本数: {len(train_dataset)} (正: {pos_count}, 负: {neg_count})")
    print(f"验证集样本数: {len(val_dataset)} (正: {np.sum(full_dataset.labels[val_indices]==1)}, 负: {np.sum(full_dataset.labels[val_indices]==0)})")
    
    # 为每个样本分配权重（平衡采样概率）
    weights = np.where(train_labels == 1, 1.0 / pos_count, 1.0 / neg_count)
    weights = torch.tensor(weights, dtype=torch.float32)
    
    # 创建加权随机采样器
    train_sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_indices),  # 采样数量=训练集大小
        replacement=True  # 允许重复采样（过采样少数类）
    )
    
    # --------------------------- 创建数据加载器 ---------------------------
    # 训练加载器：使用平衡采样器，关闭shuffle
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler,  # 使用平衡采样器
        shuffle=False,          # sampler和shuffle不能同时开启
        num_workers=4,
        pin_memory=True,
        drop_last=True          # 丢弃最后不完整的batch
    )
    
    # 验证加载器：保持原有分布，不采样
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )


    # --------------------------- 模型初始化 ---------------------------

    """
    model = LRDClassifier(
        learning_rate=LEARNING_RATE,
        npixels=NPIXELS,
        random_mask_ratio=RANDOM_MASK_RATIO,
        random_shift=RANDOM_SHIFT,
        noise_std_max=NOISE_STD_MAX,
        model_type=MODEL_TYPE,
        hidden_dims=HIDDEN_DIMS,
        rank=RANK,
        dropout_rate=DROPOUT_RATE
    )
    """

    model = LRDClassifier.load_from_checkpoint(
        '/home/zechang/user/DESI_LRD_ML/swanlab_logs/lrd_classification/lrd_lowrank_cnn_20251224_172547/checkpoints/lrd-lowrank_cnn-epoch=04-val_loss=0.3005.ckpt'
    )
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"===== 模型信息 =====")
    print(f"模型类型: {MODEL_TYPE}")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # --------------------------- 训练器配置 ---------------------------
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(SWAN_SAVE_DIR, SWAN_PROJECT, EXPERIMENT_NAME, "checkpoints"),
        filename=f'lrd-{MODEL_TYPE}-{{epoch:02d}}-{{val_loss:.4f}}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='auto',
        devices=1,
        logger=swanlogger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=1.0
    )
    
    # --------------------------- 开始训练 ---------------------------
    trainer.fit(model, train_loader, val_loader)
    
    print(f"训练完成！日志和模型保存在: {os.path.join(SWAN_SAVE_DIR, SWAN_PROJECT, EXPERIMENT_NAME)}")

if __name__ == '__main__':
    # 设置随机种子确保可复现
    pl.seed_everything(42, workers=True)
    
    # 启动训练
    main()

