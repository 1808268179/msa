import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import build_backbone


class RegionTokenizer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, feat):
        b, c, h, w = feat.shape
        pooled = F.adaptive_avg_pool2d(feat, output_size=(self.scale, self.scale))
        tokens = pooled.flatten(2).transpose(1, 2)  # [B, N, C]
        return tokens


class CrossScaleSelfAttention(nn.Module):
    def __init__(self, dim, attn_dim=512, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(dim, attn_dim)
        self.k = nn.Linear(dim, attn_dim)
        self.v = nn.Linear(dim, attn_dim)
        self.proj = nn.Linear(attn_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(attn_dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self.proj(out)
        return out, attn


class FusionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class MSANet(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone="resnet50",
        pretrained=True,
        scales=(1, 2, 4, 8),
        embed_dim=2048,
        attn_dim=512,
        dropout=0.2,
    ):
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained=pretrained)
        self.scales = scales
        self.embed_dim = self.backbone.out_channels

        self.tokenizers = nn.ModuleList([RegionTokenizer(s) for s in scales])
        self.attn = CrossScaleSelfAttention(
            self.embed_dim, attn_dim=attn_dim, dropout=dropout
        )

        self.fusion = FusionBlock(self.embed_dim * 2, self.embed_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.embed_dim, num_classes)
        )

    def forward_features(self, x):
        feat = self.backbone(x)  # [B, C, H, W]
        global_feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)

        tokens_per_scale = [tok(feat) for tok in self.tokenizers]
        all_tokens = torch.cat(tokens_per_scale, dim=1)  # [B, sum(s^2), C]
        attended_tokens, attn_map = self.attn(all_tokens)
        local_feat = attended_tokens.mean(dim=1)

        b, c, h, w = feat.shape
        global_map = global_feat.view(b, c, 1, 1).expand(-1, -1, h, w)
        local_map = local_feat.view(b, c, 1, 1).expand(-1, -1, h, w)
        fused = self.fusion(torch.cat([global_map, local_map], dim=1))
        fused_vec = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        return fused_vec, {
            "feat": feat,
            "global_feat": global_feat,
            "local_feat": local_feat,
            "attn_map": attn_map,
            "tokens": all_tokens,
        }

    def forward(self, x, return_aux=False):
        fused_vec, aux = self.forward_features(x)
        logits = self.classifier(fused_vec)
        if return_aux:
            return logits, aux
        return logits


def build_model(cfg):
    return MSANet(
        num_classes=cfg["dataset"]["num_classes"],
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"].get("pretrained", True),
        scales=tuple(cfg["model"].get("scales", [1, 2, 4, 8])),
        embed_dim=cfg["model"].get("embed_dim", 2048),
        attn_dim=cfg["model"].get("attn_dim", 512),
        dropout=cfg["model"].get("dropout", 0.2),
    )
