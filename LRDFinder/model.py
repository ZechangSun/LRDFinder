import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from typing import Tuple, Optional


class LowRankMLPLayer(nn.Module):
    def __init__(self, 
                 input_dim: int=255, 
                 output_dim: int=128,
                 rank: int=16):
        super(LowRankMLPLayer, self).__init__()

        self.rank = rank
        self.low_rank_U = nn.Linear(input_dim, rank, bias=False)
        self.low_rank_V = nn.Linear(rank, output_dim, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.low_rank_U(x)
        x = self.low_rank_V(x)
        return x


class LowRankCNN(nn.Module):
    def __init__(self, 
                 input_length=255,
                 conv_channels=[16, 32, 64],
                 fc_dims=[128, 64],
                 rank=8,
                 num_classes=1,
                 dropout_rate=0.2,
                 activation='relu'):
        super(LowRankCNN, self).__init__()
        

        self.conv_layers = nn.ModuleList()
        in_channels = 2
        
        for i, out_channels in enumerate(conv_channels):
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv_layers.append(conv)
            in_channels = out_channels

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv_output_size = conv_channels[-1] * (input_length // (2 ** len(conv_channels)))
        
        self.activation = activation
        self.fc_layers = nn.ModuleList()
        prev_dim = self.conv_output_size
        
        for fc_dim in fc_dims:
            self.fc_layers.append(LowRankMLPLayer(prev_dim, fc_dim, rank))
            self.fc_layers.append(nn.BatchNorm1d(fc_dim))
            if activation == 'relu':
                self.fc_layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                self.fc_layers.append(nn.LeakyReLU())
            else:
                raise ValueError("Unsupported activation function")
            self.fc_layers.append(nn.Dropout(dropout_rate))
            prev_dim = fc_dim
        
        self.rank = rank
        self.low_rank_U = nn.Linear(prev_dim, rank, bias=False)
        self.low_rank_V = nn.Linear(rank, num_classes, bias=True)

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = F.relu(x)
            x = self.pool(x)

        x = x.view(x.size(0), -1)

        for i in range(0, len(self.fc_layers), 4):
            x = self.fc_layers[i](x)  # Linear
            x = self.fc_layers[i+1](x)  # BatchNorm
            x = self.fc_layers[i+2](x)  # ReLU
            x = self.fc_layers[i+3](x)  # Dropout
        
        x = self.low_rank_U(x)
        x = self.low_rank_V(x)
        return x



class LowRankMLP(nn.Module):
    def __init__(self, 
                 input_dim=255, 
                 hidden_dims=[256, 128, 64],
                 rank=16,  # 低秩矩阵的秩
                 num_classes=1,
                 dropout_rate=0.2):
        super(LowRankMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.mlp_layers = nn.Sequential(*layers)
        self.rank = rank
        self.low_rank_mlp_layer = LowRankMLPLayer(input_dim=prev_dim, 
                    output_dim=num_classes, 
                    rank=rank)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.mlp_layers(x)
        x = self.low_rank_mlp_layer(x)
        return x



class MaskedAutoEncoder(nn.Module):
    def __init__(self, input_dim=255, latent_dim=64, hidden_dims=[128, 256, 128], 
                 dropout_rate=0.1, use_batch_norm=True):
        super(MaskedAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 构建编码器
        encoder_layers = []
        prev_dim = input_dim
        
        for i, h_dim in enumerate(hidden_dims):
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        if use_batch_norm:
            encoder_layers.append(nn.BatchNorm1d(latent_dim))
        encoder_layers.append(nn.Tanh())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        decoder_hidden = hidden_dims[::-1]
        
        prev_dim = latent_dim
        
        for i, h_dim in enumerate(decoder_hidden):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x, mask=None):
        batch_size = x.shape[0]

        if mask is None:
            mask = torch.bernoulli(torch.full_like(x, 0.8))

        mask = mask.to(x.device)
        x_masked = x * mask
        z = self.encoder(x_masked)
        x_recon = self.decoder(z)
        if self.training:
            return x_recon, z, mask
        else:
            x_complete = x * mask + x_recon * (1 - mask)
            return x_complete, z, mask
    
    def encode(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x)
        x_masked = x * mask.to(x.device)
        return self.encoder(x_masked)
    
    def decode(self, z):
        return self.decoder(z)



class OHEMBCELoss(nn.Module):
    def __init__(self, ohem_ratio: float = 0.2, weight: torch.Tensor = None,
                 size_average=None, reduce=None, reduction: str = 'mean'):
        super(OHEMBCELoss, self).__init__()
        if size_average is not None or reduce is not None:
            reduction = nn._reduction.default_reduction(size_average, reduce)
        if not (0 < ohem_ratio <= 1.0):
            raise ValueError(f"ohem_ratio must be in (0, 1], got {ohem_ratio}")
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
        self.bce_loss = nn.BCELoss(weight=weight, reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        loss = self.bce_loss(input, target)
        loss_flat = loss.flatten()
        batch_size = loss_flat.numel()
        num_hard = max(1, int(batch_size * self.ohem_ratio))
        _, hard_indices = torch.topk(loss_flat, num_hard, largest=True)
        hard_loss = loss_flat[hard_indices]
        if self.reduction == 'none':
            return hard_loss
        elif self.reduction == 'mean':
            return hard_loss.mean()
        elif self.reduction == 'sum':
            return hard_loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class OHEMBCEWithLogitsLoss(nn.Module):
    def __init__(self, ohem_ratio: float = 0.5, weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None, reduction: str = 'mean'):
        super(OHEMBCEWithLogitsLoss, self).__init__()

        if not (0 < ohem_ratio <= 1.0):
            raise ValueError(f"ohem_ratio must be in (0, 1], got {ohem_ratio}")
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
        

        self.bce_logits_loss = nn.BCEWithLogitsLoss(
            weight=weight, pos_weight=pos_weight, reduction='none'
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.bce_logits_loss(input, target)
        
        loss_flat = loss.flatten()
        batch_size = loss_flat.numel()
        
        num_hard = max(1, int(batch_size * self.ohem_ratio))
        
        _, hard_indices = torch.topk(loss_flat, num_hard, largest=True)
        hard_loss = loss_flat[hard_indices]

        if self.reduction == 'none':
            return hard_loss
        elif self.reduction == 'mean':
            return hard_loss.mean()
        elif self.reduction == 'sum':
            return hard_loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")



class RotaryPositionEMbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        seq = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i , j -> i j', seq, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)
    
    def forward(self, x: torch.Tensor, seq_dim: int = -2):
        seq_len = x.shape[seq_dim]

        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len * 2
            self._build_cache(self.max_seq_len)
        
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        x1, x2 = x.chunk(2, dim=-1)

        rotated = torch.cat((-x2, x1), dim=-1)

        return (x * cos) + (rotated * sin)


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** (-0.5)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.rope = RotaryPositionEMbedding(dim=self.head_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), 
                float('-inf')
            )
        
        attn_output = torch.matmul(attn_probs, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )

        return self.out_proj(attn_output)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
        dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()

        self.self_attn = MultiHeadAttentionWithRoPE(
            d_model, nhead, dropout
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == 'relu' else F.gelu
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        pass





