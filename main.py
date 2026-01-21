import os
import numpy as np
import argparse
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from LRDFinder.lightning import LRDFinder
from LRDFinder.dataset import LRDDataset, PairWiseLRDDataset

def parse_args():
    parser = argparse.ArgumentParser(description="LRDFinder Training and Inference")
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                        help='Mode: train or inference')
    
    # Data paths
    parser.add_argument('--pos_fits_files', type=str, nargs='+', default=[],
                        help='List of positive FITS files')
    parser.add_argument('--neg_fits_files', type=str, nargs='+', default=[],
                        help='List of negative FITS files')
    parser.add_argument('--inference_files', type=str, nargs='+', default=[],
                        help='List of FITS files for inference')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='lowrank_cnn', choices=['lowrank_cnn', 'lowrank_mlp'],
                        help='Model type')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128, 64],
                        help='Hidden dimensions for MLP')
    parser.add_argument('--rank', type=int, default=16,
                        help='Rank for low-rank layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'rank'],
                        help='Task type: classification or rank')
    
    # Data processing
    parser.add_argument('--npixels', type=int, default=255,
                        help='Number of pixels per spectrum')
    parser.add_argument('--random_mask_ratio', type=float, default=0.1,
                        help='Random mask ratio')
    parser.add_argument('--random_shift', type=int, nargs=2, default=[-25, 75],
                        help='Random shift range')
    parser.add_argument('--noise_std_max', type=float, default=0.05,
                        help='Maximum noise standard deviation')
    
    # Checkpoint and logging
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint file for inference or resuming training')
    parser.add_argument('--swan_project', type=str, default='lrd_finder',
                        help='SwanLab project name')
    parser.add_argument('--swan_save_dir', type=str, default='./swanlab_logs',
                        help='SwanLab save directory')
    
    return parser.parse_args()

def prepare_data(args):
    """Prepare datasets and dataloaders based on task type"""
    if args.task == 'classification':
        full_dataset = LRDDataset(
            pos_fits_files=args.pos_fits_files,
            neg_fits_files=args.neg_fits_files,
            npixels=args.npixels,
            random_mask_ratio=args.random_mask_ratio,
            random_shift=args.random_shift,
            noise_std_max=args.noise_std_max,
            is_train=True
        )
    elif args.task == 'rank':
        full_dataset = PairWiseLRDDataset(
            pos_fits_files=args.pos_fits_files,
            neg_fits_files=args.neg_fits_files,
            npixels=args.npixels,
            random_mask_ratio=args.random_mask_ratio,
            random_shift=args.random_shift,
            noise_std_max=args.noise_std_max,
            is_train=True
        )
    else:
        raise ValueError(f"Unknown task type: {args.task}")
    
    # Split dataset
    pos_idx = np.where(full_dataset.labels == 1)[0]
    neg_idx = np.where(full_dataset.labels == 0)[0]
    
    np.random.seed(42)
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    # Create validation set with balanced classes
    val_target_size = 2000
    val_per_class = val_target_size // 2
    
    val_pos_num = min(val_per_class, len(pos_idx))
    val_neg_num = min(val_per_class, len(neg_idx))
    
    val_pos_idx = pos_idx[:val_pos_num]
    val_neg_idx = neg_idx[:val_neg_num]
    val_indices = np.concatenate([val_pos_idx, val_neg_idx])
    
    train_pos_idx = pos_idx[val_pos_num:]
    train_neg_idx = neg_idx[val_neg_num:]
    train_indices = np.concatenate([train_pos_idx, train_neg_idx])
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Create weighted sampler for training
    train_labels = full_dataset.labels[train_indices]
    pos_count = np.sum(train_labels == 1)
    neg_count = len(train_labels) - pos_count
    
    weights = np.where(train_labels == 1, 1.0 / pos_count, 1.0 / neg_count)
    weights = torch.tensor(weights, dtype=torch.float32)
    
    train_sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_indices),
        replacement=True
    )
    
    # Create dataloaders
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
    
    # Print dataset statistics
    print(f"===== 数据集统计 ====")
    print(f"总样本数: {len(full_dataset)} (正: {np.sum(full_dataset.labels==1)}, 负: {np.sum(full_dataset.labels==0)})")
    print(f"训练集样本数: {len(train_dataset)} (正: {pos_count}, 负: {neg_count})")
    print(f"验证集样本数: {len(val_dataset)} (正: {val_pos_num}, 负: {val_neg_num})")
    
    return train_loader, val_loader

def prepare_inference_data(args):
    """Prepare dataset for inference"""
    # Use LRDDataset with is_train=False for inference
    inference_dataset = LRDDataset(
        pos_fits_files=args.inference_files,  # Use inference_files as positive files
        neg_fits_files=[],  # No negative files for inference
        npixels=args.npixels,
        random_mask_ratio=0.0,  # No masking during inference
        random_shift=(0, 0),  # No shifting during inference
        noise_std_max=0.0,  # No noise during inference
        is_train=False
    )
    
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"===== 推理数据集统计 ====")
    print(f"推理样本数: {len(inference_dataset)}")
    
    return inference_loader

def train(args):
    """Train the model"""
    # Prepare data
    train_loader, val_loader = prepare_data(args)
    
    # Initialize model
    if args.checkpoint_path:
        # Resume from checkpoint
        model = LRDFinder.load_from_checkpoint(args.checkpoint_path)
        print(f"从检查点恢复: {args.checkpoint_path}")
    else:
        # Initialize new model
        model = LRDFinder(
            learning_rate=args.learning_rate,
            npixels=args.npixels,
            random_mask_ratio=args.random_mask_ratio,
            random_shift=args.random_shift,
            noise_std_max=args.noise_std_max,
            model_type=args.model_type,
            hidden_dims=args.hidden_dims,
            rank=args.rank,
            dropout_rate=args.dropout_rate,
            task=args.task
        )
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"===== 模型信息 ====")
    print(f"模型类型: {args.model_type}")
    print(f"任务类型: {args.task}")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # Create experiment name
    experiment_name = f"lrd_{args.model_type}_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Configure checkpoint callback
    checkpoint_dir = os.path.join(args.swan_save_dir, args.swan_project, experiment_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename=f'lrd-{args.model_type}-{{epoch:02d}}-{{val_loss:.4f}}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    
    # Configure trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=1.0
    )
    
    # Start training
    print("===== 开始训练 ====")
    trainer.fit(model, train_loader, val_loader)
    
    print(f"训练完成！模型保存在: {checkpoint_dir}")

def inference(args):
    """Run inference on new data"""
    if not args.checkpoint_path:
        raise ValueError("必须提供检查点路径用于推理")
    
    if not args.inference_files:
        raise ValueError("必须提供推理数据文件")
    
    # Prepare data
    inference_loader = prepare_inference_data(args)
    
    # Load model from checkpoint
    model = LRDFinder.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    print(f"从检查点加载模型: {args.checkpoint_path}")
    
    # Run inference
    print("===== 开始推理 ====")
    
    all_preds = []
    all_labels = []
    all_radecs = []
    
    with torch.no_grad():
        for batch in inference_loader:
            # Get predictions
            y_hat = model(batch).squeeze()
            
            # Apply sigmoid for classification
            if args.task == 'classification':
                y_hat = torch.sigmoid(y_hat)
            
            # Collect results
            all_preds.append(y_hat.cpu().numpy())
            if 'label' in batch:
                all_labels.append(batch['label'].cpu().numpy())
            if 'radec' in batch:
                all_radecs.append(batch['radec'].cpu().numpy())
    
    # Concatenate results
    all_preds = np.concatenate(all_preds)
    if all_labels:
        all_labels = np.concatenate(all_labels)
    if all_radecs:
        all_radecs = np.concatenate(all_radecs)
    
    # Save inference results
    output_dir = './inference_results'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
    save_dict = {'preds': all_preds}
    
    if all_labels:
        save_dict['labels'] = all_labels
    if all_radecs:
        save_dict['radecs'] = all_radecs
    
    np.savez(output_file, **save_dict)
    print(f"推理结果保存到: {output_file}")
    
    # Print inference summary
    print(f"===== 推理结果 ====")
    print(f"预测样本数: {len(all_preds)}")
    print(f"预测值范围: [{all_preds.min():.4f}, {all_preds.max():.4f}]")
    print(f"预测值均值: {all_preds.mean():.4f}")
    print(f"预测值标准差: {all_preds.std():.4f}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Create necessary directories
    os.makedirs(args.swan_save_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Run training or inference
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    main()