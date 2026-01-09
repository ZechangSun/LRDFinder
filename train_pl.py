import os
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
import argparse

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from LRDFinder.lightning import LRDFinder
from LRDFinder.dataset import LRDDataset

import swanlab
from swanlab.integration.pytorch_lightning import SwanLabLogger


# 设置SwanLab根目录
SWANLAB_ROOT = "/home/zechang/user/DESI_LRD_ML/swanlab_data"
os.environ["SWANLAB_SAVE_DIR"] = SWANLAB_ROOT 
os.makedirs(SWANLAB_ROOT, exist_ok=True)



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LRD分类模型训练脚本')
    
    # 数据文件参数
    parser.add_argument('--pos_fits_files', nargs='+', default=['/home/zechang/user/DESI_LRD_ML/data/mock_spectra_Ha_with_abs.fits'],
                        help='正样本FITS文件路径列表')
    parser.add_argument('--neg_fits_files', nargs='+', default=['/home/zechang/user/DESI_LRD_ML/data/real_spectra_Hb_psf_100000.fits'],
                        help='负样本FITS文件路径列表')
    
    # 模型超参数
    parser.add_argument('--model_type', type=str, default='lowrank_cnn', choices=['lowrank_cnn', 'lowrank_mlp'],
                        help='模型类型')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256, 128, 64],
                        help='隐藏层维度列表')
    parser.add_argument('--rank', type=int, default=128,
                        help='低秩分解的秩')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='dropout率')
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'rank'],
                        help='任务类型')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='最大训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='学习率')
    parser.add_argument('--npixels', type=int, default=255,
                        help='光谱像素数')
    parser.add_argument('--random_mask_ratio', type=float, default=0.1,
                        help='随机掩码比例')
    parser.add_argument('--random_shift', nargs=2, type=int, default=(-25, 75),
                        help='随机偏移范围')
    parser.add_argument('--noise_std_max', type=float, default=0.05,
                        help='最大噪声标准差')
    
    # 验证集参数
    parser.add_argument('--val_target_size', type=int, default=2000,
                        help='验证集总大小')
    
    # SwanLab参数
    parser.add_argument('--swan_project', type=str, default='lrd_classification',
                        help='SwanLab项目名称')
    parser.add_argument('--swan_save_dir', type=str, default='/home/zechang/user/DESI_LRD_ML/swanlab_logs',
                        help='SwanLab日志保存目录')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='实验名称，默认自动生成')
    
    # 模型加载参数
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='预训练模型检查点路径')
    
    # 随机种子
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()


def create_dataset(args):
    """创建数据集并拆分训练集和验证集"""
    # 创建完整数据集
    full_dataset = LRDDataset(
        pos_fits_files=args.pos_fits_files,
        neg_fits_files=args.neg_fits_files,
        npixels=args.npixels,
        random_mask_ratio=args.random_mask_ratio,
        random_shift=tuple(args.random_shift),
        noise_std_max=args.noise_std_max,
        is_train=True
    )
    
    # 分离正/负样本的索引
    pos_idx = np.where(full_dataset.labels == 1)[0]
    neg_idx = np.where(full_dataset.labels == 0)[0]
    
    # 设置随机种子，保证结果可复现
    np.random.seed(args.seed)
    
    # 定义验证集每类目标数量
    val_per_class = args.val_target_size // 2
    
    # 确保每类样本数不超过实际存在的数量
    val_pos_num = min(val_per_class, len(pos_idx))
    val_neg_num = min(val_per_class, len(neg_idx))
    
    # 随机打乱正负样本索引
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    # 拆分验证集/训练集的正负样本索引
    val_pos_idx = pos_idx[:val_pos_num]
    val_neg_idx = neg_idx[:val_neg_num]
    val_indices = np.concatenate([val_pos_idx, val_neg_idx])
    
    train_pos_idx = pos_idx[val_pos_num:]
    train_neg_idx = neg_idx[val_neg_num:]
    train_indices = np.concatenate([train_pos_idx, train_neg_idx])
    
    # 创建子集
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    return train_dataset, val_dataset, full_dataset, train_indices, val_indices


def create_dataloaders(train_dataset, val_dataset, train_indices, full_dataset, args):
    """创建数据加载器"""
    # 统计并打印信息
    train_labels = full_dataset.labels[train_indices]
    pos_count = np.sum(train_labels == 1)
    neg_count = len(train_labels) - pos_count
    
    val_labels = full_dataset.labels[val_dataset.indices]
    val_pos_count = np.sum(val_labels == 1)
    val_neg_count = len(val_labels) - val_pos_count
    
    print(f"===== 数据集统计 =====")
    print(f"总样本数: {len(full_dataset)} (正: {np.sum(full_dataset.labels==1)}, 负: {np.sum(full_dataset.labels==0)})")
    print(f"训练集样本数: {len(train_dataset)} (正: {pos_count}, 负: {neg_count})")
    print(f"验证集样本数: {len(val_dataset)} (正: {val_pos_count}, 负: {val_neg_count})")
    
    # 为每个样本分配权重（平衡采样概率）
    weights = np.where(train_labels == 1, 1.0 / pos_count, 1.0 / neg_count)
    weights = torch.tensor(weights, dtype=torch.float32)
    
    # 创建加权随机采样器
    train_sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_indices),
        replacement=True
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_model(args):
    """创建模型"""
    if args.checkpoint_path:
        # 从检查点加载模型
        model = LRDFinder.load_from_checkpoint(args.checkpoint_path)
        print(f"从检查点加载模型: {args.checkpoint_path}")
    else:
        # 初始化新模型
        model = LRDFinder(
            learning_rate=args.learning_rate,
            npixels=args.npixels,
            random_mask_ratio=args.random_mask_ratio,
            random_shift=tuple(args.random_shift),
            noise_std_max=args.noise_std_max,
            model_type=args.model_type,
            hidden_dims=args.hidden_dims,
            rank=args.rank,
            dropout_rate=args.dropout_rate,
            task=args.task
        )
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"===== 模型信息 =====")
    print(f"模型类型: {args.model_type}")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    return model


def main(args=None):
    """主函数"""
    # 解析参数
    if args is None:
        args = parse_args()
    
    # 自动生成实验名称（如果未提供）
    if args.experiment_name is None:
        args.experiment_name = f"lrd_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 初始化SwanLab日志器
    swanlogger = SwanLabLogger(
        project=args.swan_project,
        experiment_name=args.experiment_name,
        save_dir=args.swan_save_dir,
        api_key="kSQ0WETobYCojrxzRkZCX"
    )
    
    # 更新SwanLab配置
    swanlab.config.update({
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "npixels": args.npixels,
        "random_mask_ratio": args.random_mask_ratio,
        "random_shift": tuple(args.random_shift),
        "noise_std_max": args.noise_std_max,
        "model_type": args.model_type,
        "hidden_dims": args.hidden_dims,
        "rank": args.rank,
        "dropout_rate": args.dropout_rate,
        "task": args.task,
        "sampling_strategy": "weighted_random_sampler"
    })
    
    # 创建数据集
    train_dataset, val_dataset, full_dataset, train_indices, val_indices = create_dataset(args)
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, train_indices, full_dataset, args)
    
    # 创建模型
    model = create_model(args)
    
    # 配置检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(args.swan_save_dir, args.swan_project, args.experiment_name, "checkpoints"),
        filename=f'lrd-{args.model_type}-{{epoch:02d}}-{{val_loss:.4f}}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    
    # 初始化训练器
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=1,
        logger=swanlogger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=1.0
    )
    
    # 开始训练
    trainer.fit(model, train_loader, val_loader)
    
    print(f"训练完成！日志和模型保存在: {os.path.join(args.swan_save_dir, args.swan_project, args.experiment_name)}")
    
    return os.path.join(args.swan_save_dir, args.swan_project, args.experiment_name)

if __name__ == '__main__':
    # 设置随机种子确保可复现
    pl.seed_everything(42, workers=True)
    
    # 启动训练
    main()

