"""
模型定义：BirdCLEF-B0 基线 + SED 模型
基于 DES-001 §二
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from . import config as CFG


class GeM(nn.Module):
    """Generalized Mean Pooling (p=3.0)"""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W) → (B, C)
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p).squeeze(-1).squeeze(-1)


class BirdCLEFB0(nn.Module):
    """
    EfficientNet-B0 + GeM Pooling + 辅助特征
    Input:  mel (B, 1, 128, T),  insect_energy (B, 1),  hour (B,) optional
    Output: logits (B, num_classes)
    """

    def __init__(self, num_classes: int = CFG.NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b0_ns", pretrained=pretrained, in_chans=1,
            features_only=False, num_classes=0, global_pool=""
        )
        self.gem = GeM(p=3.0)
        backbone_out = 1280  # EfficientNet-B0 last feature dim

        self.hour_embed = nn.Embedding(24, 32)
        aux_dim = 32 + 1  # hour(32) + insect_energy(1)

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(backbone_out, 512)
        self.fc2 = nn.Linear(512 + aux_dim, num_classes)

    def forward(self, mel, insect_energy=None, hour=None):
        # mel: (B, 1, 128, T)
        feat = self.backbone(mel)             # (B, 1280, H, W)
        feat = self.gem(feat)                 # (B, 1280)
        feat = self.dropout(feat)
        feat = F.relu(self.fc1(feat))         # (B, 512)

        aux_parts = []
        if hour is not None:
            aux_parts.append(self.hour_embed(hour))  # (B, 32)
        else:
            aux_parts.append(torch.zeros(feat.size(0), 32, device=feat.device))
        if insect_energy is not None:
            aux_parts.append(insect_energy)          # (B, 1)
        else:
            aux_parts.append(torch.zeros(feat.size(0), 1, device=feat.device))

        aux = torch.cat(aux_parts, dim=-1)    # (B, 33)
        out = torch.cat([feat, aux], dim=-1)  # (B, 545)
        logits = self.fc2(out)                # (B, 234)
        return logits


class SEDB0(nn.Module):
    """
    Sound Event Detection 模型（Stage 3+）
    帧级 Attention Pooling
    用 forward_features 获取 conv_head 后的 1280-d 特征
    """

    def __init__(self, num_classes: int = CFG.NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b0_ns", pretrained=pretrained, in_chans=1,
            num_classes=0, global_pool=""
        )
        backbone_out = 1280

        self.att_fc = nn.Linear(backbone_out, num_classes)
        self.cls_fc = nn.Linear(backbone_out, num_classes)

    def forward(self, mel, **kwargs):
        # mel: (B, 1, 128, T)
        features = self.backbone(mel)          # (B, 1280, H, W)
        B, C, H, W = features.shape
        features = features.reshape(B, C, H * W)  # (B, 1280, T')
        features = features.permute(0, 2, 1)      # (B, T', 1280)

        att = torch.sigmoid(self.att_fc(features))      # (B, T', num_classes)
        framewise = torch.sigmoid(self.cls_fc(features)) # (B, T', num_classes)

        clipwise = (att * framewise).sum(dim=1) / (att.sum(dim=1) + 1e-6)
        return clipwise


class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification"""

    def __init__(self, gamma: float = CFG.FOCAL_GAMMA, alpha: float = CFG.FOCAL_ALPHA):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class BirdCLEFLoss(nn.Module):
    """BCE + Focal 混合损失"""

    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction="mean"
        )
        self.focal = FocalLoss()

    def forward(self, logits, targets):
        return CFG.BCE_WEIGHT * self.bce(logits, targets) \
             + CFG.FOCAL_WEIGHT * self.focal(logits, targets)
