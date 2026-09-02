import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScaleMotifCNN(nn.Module):
    def __init__(self, vocab_size=40, dim=480):
        super(MultiScaleMotifCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=1)
        d1 = dim // 3
        d2 = dim // 3
        d3 = dim - d1 - d2
        self.conv3 = nn.Conv1d(dim, d1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(dim, d2, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(dim, d3, kernel_size=7, padding=3)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_ids):
        x = self.embed(input_ids).transpose(1, 2)
        c3 = self.act(self.conv3(x))
        c5 = self.act(self.conv5(x))
        c7 = self.act(self.conv7(x))
        out = torch.cat([c3, c5, c7], dim=1).transpose(1, 2)
        return self.norm(out)

class AttentionPooling(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super(AttentionPooling, self).__init__()
        self.key_layer = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh()
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        keys = self.key_layer(x)
        scores = torch.matmul(keys, self.query.transpose(1, 2)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)
        gated_x = x * self.gate(x)
        global_feat = torch.sum(attn_weights * gated_x, dim=1)
        return global_feat

class RDBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super(RDBlock, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x0 = x
        x = self.norm(x)
        x = self.act(self.dense(x))
        x = self.dropout(x)
        x = x0 + x
        return x

class GeometricGNN(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super(GeometricGNN, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(dim + hidden_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, dist_matrix, mask=None):
        edge_feats = self.edge_mlp(dist_matrix.unsqueeze(-1))
        if mask is not None:
            edge_feats = edge_feats * mask.unsqueeze(-1)
        aggregated_messages = torch.sum(edge_feats, dim=2)
        cat_feats = torch.cat([h, aggregated_messages], dim=-1)
        out = self.node_mlp(cat_feats)
        return self.norm(h + out)

class TGCFusion(nn.Module):
    def __init__(self, dim, n_heads=4):
        super(TGCFusion, self).__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.dist_bias_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, n_heads)
        )
        self.diff_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, seq_emb, struct_emb, dist_matrix, mask=None):
        B, L, D = seq_emb.shape
        geo_bias = self.dist_bias_net(dist_matrix.unsqueeze(-1))
        geo_bias = geo_bias.permute(0, 3, 1, 2)

        q = self.q_proj(seq_emb).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(struct_emb).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(struct_emb).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores + geo_bias

        if mask is not None:
            mask_bool = (mask == 0).view(B, 1, 1, L)
            attn_scores = attn_scores.masked_fill(mask_bool, -1e4)

        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v).transpose(1, 2).reshape(B, L, D)

        diff_feat = (seq_emb - struct_emb).pow(2)
        calibration_gate = self.diff_proj(diff_feat)

        fused = seq_emb + calibration_gate * context
        return self.norm(self.out_proj(fused))

class MultimodalEnzymeModel(nn.Module):
    def __init__(self, dim=480, n_layers=2, n_RD=3):
        super(MultimodalEnzymeModel, self).__init__()
        self.local_cnn = MultiScaleMotifCNN(vocab_size=40, dim=dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.struct_encoder = nn.ModuleList([
            GeometricGNN(dim, hidden_dim=64) for _ in range(n_layers)
        ])
        self.fusion = TGCFusion(dim, n_heads=4)
        self.pooler = AttentionPooling(dim, hidden_dim=128)
        self.rds = nn.ModuleList([RDBlock(dim) for _ in range(n_RD)])
        self.output_head = nn.Linear(dim, 1)

    def forward(self, input_ids, seq_emb, coords, mask=None):
        cnn_feat = self.local_cnn(input_ids)
        gate = self.fusion_gate(torch.cat([seq_emb, cnn_feat], dim=-1))
        fused_seq_emb = seq_emb + gate * cnn_feat

        dist_matrix = torch.cdist(coords, coords, p=2) / 10.0
        spatial_mask = (dist_matrix < 1.5).float()
        if mask is not None:
            mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
            spatial_mask = spatial_mask * mask_2d

        struct_feat = fused_seq_emb
        for layer in self.struct_encoder:
            struct_feat = layer(struct_feat, dist_matrix, spatial_mask)

        fused_feat = self.fusion(fused_seq_emb, struct_feat, dist_matrix, mask)
        global_feat = self.pooler(fused_feat, mask)

        x = global_feat
        for rd in self.rds:
            x = rd(x)
        return self.output_head(x)