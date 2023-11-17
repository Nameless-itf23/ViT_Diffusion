# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VitInputLayer(nn.Module): 
    def __init__(self, in_channels:int=3, emb_dim:int=192, num_patch_row:int=4, image_size:int=32, t_max:int=500):
        super(VitInputLayer, self).__init__() 
        self.in_channels=in_channels 
        self.emb_dim = emb_dim 
        self.num_patch_row = num_patch_row 
        self.image_size = image_size
        
        # number of patch
        self.num_patch = self.num_patch_row**2

        # size of patch
        self.patch_size = int(self.image_size // self.num_patch_row)

        # make patch
        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=self.emb_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )

        # posisional encoding for time embedding
        position = torch.arange(t_max).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        pe = torch.zeros(t_max, emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # positional embedding
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.num_patch+1, emb_dim) 
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, D, H/P, W/P)
        z_0 = self.patch_emb_layer(x)

        # (B, D, H/P, W/P) -> (B, D, Np) 
        # Np: patch (=H*W/Pˆ2)
        z_0 = z_0.flatten(2)

        # (B, D, Np) -> (B, Np, D) 
        z_0 = z_0.transpose(1, 2)

        # positional encoding
        z_0 = torch.cat(
            [self.pe[t], z_0], dim=1)

        # positional embedding
        # (B, N, D) -> (B, N, D) 
        z_0 = z_0 + self.pos_emb
        return z_0


class MultiHeadSelfAttention(nn.Module): 
    def __init__(self, emb_dim:int=192, head:int=3, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__() 
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim**0.5

        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout) 
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        batch_size, num_patch, _ = z.size()

        # (B, N, D) -> (B, N, D)
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # (B, N, D) -> (B, N, h, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        # (B, N, h, D//h) -> (B, h, N, D//h)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # (B, h, N, D//h) -> (B, h, D//h, N)
        k_T = k.transpose(2, 3)

        # (B, h, N, D//h) x (B, h, D//h, N) -> (B, h, N, N) 
        dots = (q @ k_T) / self.sqrt_dh
        attn = F.softmax(dots, dim=-1)
        attn = self.attn_drop(attn)

        # (B, h, N, N) x (B, h, N, D//h) -> (B, h, N, D//h) 
        out = attn @ v

        # (B, h, N, D//h) -> (B, N, h, D//h)
        out = out.transpose(1, 2)

        # (B, N, h, D//h) -> (B, N, D)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        # (B, N, D) -> (B, N, D) 
        out = self.w_o(out) 
        return out


class VitEncoderBlock(nn.Module): 
    def __init__(self, emb_dim:int=192, head:int=8, hidden_dim:int=192*4, dropout: float=0.):
        super(VitEncoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.msa = MultiHeadSelfAttention(
        emb_dim=emb_dim, head=head,
        dropout = dropout,
        )
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential( 
            nn.Linear(emb_dim, hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, emb_dim), 
            nn.Dropout(dropout)
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.msa(self.ln1(z)) + z
        out = self.mlp(self.ln2(out)) + out 
        return out


class Vit(nn.Module): 
    def __init__(self, in_channels:int=3, emb_dim:int=192, num_patch_row:int=4, image_size:int=32, num_blocks:int=7, head:int=8, hidden_dim:int=192*4, dropout:float=0., t_max:int=500):
        super(Vit, self).__init__()
        self.input_layer = VitInputLayer(
            in_channels, 
            emb_dim, 
            num_patch_row, 
            image_size,
            t_max)

        self.encoder = nn.Sequential(*[
            VitEncoderBlock(
                emb_dim=emb_dim,
                head=head,
                hidden_dim=hidden_dim,
                dropout = dropout
            )
            for _ in range(num_blocks)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, image_size**2*in_channels//num_patch_row**2)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, N, D)
        out = self.input_layer(x)
        
        # (B, N, D) -> (B, N, D)
        out = self.encoder(out)

        # (B, N, D) -> (B, D)
        out = out[:, 1:]

        # (B, Np, D) -> (B, Np, M)
        out = self.mlp_head(out)

        # (B, Np, M) -> (B, C, H, W)
        # ???
        out = torch.reshape(out, (out.shape[0], out.shape[1], self.in_channels, -1))
        out = torch.transpose(out, 1, 2)
        pred = torch.reshape(out, (out.shape[0], self.in_channels, self.image_size, self.image_size))
        return pred
