'''
=================================================
coding:utf-8
@File:      FastGatedConformer.py
@Function: Fast gated temporal mixer for cross-subject EEG decoding
=================================================
'''

import torch
from torch import nn

from models.DBConformer import ClassificationHead, Gate_FC, PatchEmbeddingSpatial, PatchEmbeddingTemporal


class FastTemporalMixerBlock(nn.Module):
    def __init__(self, emb_size, kernel_size=3, drop_p=0.5):
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = nn.LayerNorm(emb_size)
        self.in_proj = nn.Linear(emb_size, emb_size * 2)
        self.dwconv = nn.Conv1d(
            emb_size,
            emb_size,
            kernel_size=kernel_size,
            padding=padding,
            groups=emb_size,
            bias=False,
        )
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(emb_size, emb_size)
        self.drop1 = nn.Dropout(drop_p)

    def forward(self, x):
        z, gate = self.in_proj(self.norm1(x)).chunk(2, dim=-1)
        z = z.transpose(1, 2)
        z = self.dwconv(z).transpose(1, 2)
        z = self.act(z) * torch.sigmoid(gate)
        x = x + self.drop1(self.out_proj(z))
        return x


class FastTemporalMixer(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[FastTemporalMixerBlock(emb_size) for _ in range(depth)])


class FastGatedConformer(nn.Module):
    def __init__(self, args, emb_size=40, tem_depth=2, chn_depth=2, chn=22, n_classes=2):
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

        self.temporal_mixer = FastTemporalMixer(tem_depth, emb_size)

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

        x_temporal = self.temporal_mixer(x_embed)
        x_spatial = x_embed_spatial

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
