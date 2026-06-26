'''
=================================================
coding:utf-8
@File:      DGMambaConformer.py
@Function: Domain-generalized Mamba-Conformer for cross-subject EEG decoding
=================================================
'''

import math
import torch
import torch.nn.functional as F
from einops import repeat, einsum
from torch import nn

from models.DBConformer import (
    ClassificationHead,
    Gate_FC,
    PatchEmbeddingSpatial,
    PatchEmbeddingTemporal,
    TransformerEncoder,
)


class MambaBlock(nn.Module):
    def __init__(self, input_channels, d_state=16):
        super().__init__()
        self.d_model = input_channels
        self.d_inner = self.d_model * 2
        self.dt_rank = math.ceil(self.d_model / 16)
        self.d_state = d_state

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            groups=self.d_inner,
            padding=2,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(self, x):
        b, l, _ = x.shape
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]
        x = x.transpose(1, 2)
        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        d_in, n = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        b, l, d_in = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d, d n -> b l d n'))
        deltaB_u = einsum(delta, B, u, 'b l d, b l n, b l d -> b l d n')

        x = torch.zeros((b, d_in, n), device=u.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d n, b n -> b d')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        return y + u * D


class MambaEncoderBlock(nn.Module):
    def __init__(self, emb_size, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.mamba = MambaBlock(emb_size)
        self.drop1 = nn.Dropout(drop_p)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Dropout(forward_drop_p),
            nn.Linear(forward_expansion * emb_size, emb_size),
        )
        self.drop2 = nn.Dropout(drop_p)

    def forward(self, x):
        x = x + self.drop1(self.mamba(self.norm1(x)))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class MambaEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[MambaEncoderBlock(emb_size) for _ in range(depth)])


class DGMambaConformer(nn.Module):
    def __init__(self, args, emb_size=40, tem_depth=5, chn_depth=5, chn=22, n_classes=2):
        super().__init__()
        self.embedding = PatchEmbeddingTemporal(
            data_name=args.data_name,
            in_planes=args.chn,
            out_planes=emb_size,
            kernel_size=63,
            radix=1,
            patch_size=args.patch_size,
            time_points=args.time_sample_num,
            num_classes=args.class_num
        )
        self.channel_embedding = PatchEmbeddingSpatial(spa_dim=args.spa_dim, emb_size=emb_size)
        self.P = args.time_sample_num // args.patch_size
        self.C = args.chn
        self.D = emb_size
        self.gate_flag = args.gate_flag
        self.posemb_flag = args.posemb_flag
        self.branch = args.branch
        self.chn_atten_flag = args.chn_atten_flag

        if args.posemb_flag:
            self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.P, self.D))
            self.pos_embedding_spatial = nn.Parameter(torch.randn(1, self.C, self.D))

        self.temporal_mamba = MambaEncoder(tem_depth, emb_size)
        self.spatial_transformer = TransformerEncoder(chn_depth, emb_size)

        if args.gate_flag or self.branch == 'temporal' or self.branch == 'spatial':
            self.gate_fc = Gate_FC(emb_size)
            self.classifier = ClassificationHead(emb_size, n_classes)
        else:
            self.classifier = ClassificationHead(emb_size * 2, n_classes)
            if args.chn_atten_flag:
                self.spatial_attn_pool = nn.Sequential(
                    nn.Linear(emb_size, emb_size),
                    nn.Tanh(),
                    nn.Linear(emb_size, 1),
                )

    def forward(self, x):
        x = x.squeeze(1)
        x_embed = self.embedding(x)
        x_embed_spatial = self.channel_embedding(x)
        if self.posemb_flag:
            x_embed = x_embed + self.pos_embedding_temporal
            x_embed_spatial = x_embed_spatial + self.pos_embedding_spatial

        x_temporal = self.temporal_mamba(x_embed)
        x_spatial = self.spatial_transformer(x_embed_spatial)

        if self.branch == 'temporal':
            x_fused = x_temporal.mean(dim=1)
        elif self.branch == 'spatial':
            x_fused = x_spatial.mean(dim=1)
        elif self.gate_flag:
            gate = torch.sigmoid(
                self.gate_fc(torch.cat([x_temporal.mean(dim=1), x_spatial.mean(dim=1)], dim=-1)))
            x_fused = gate * x_spatial.mean(dim=1) + (1 - gate) * x_temporal.mean(dim=1)
        elif self.chn_atten_flag:
            x_t = x_temporal.mean(dim=1)
            attn_scores = self.spatial_attn_pool(x_spatial)
            attn_weights = torch.softmax(attn_scores, dim=1)
            x_s = torch.sum(attn_weights * x_spatial, dim=1)
            x_fused = torch.cat([x_t, x_s], dim=-1)
        else:
            x_fused = torch.cat([x_temporal.mean(dim=1), x_spatial.mean(dim=1)], dim=-1)

        _, out = self.classifier(x_fused)
        return x_fused, out
