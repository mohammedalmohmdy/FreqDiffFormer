
import torch
import torch.nn as nn
try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

class SpatialEncoder(nn.Module):
    def __init__(self, backbone_name='swin_tiny_patch4_window7_224', out_dim=256, pretrained=True):
        super().__init__()
        if _HAS_TIMM:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, features_only=True)
            feat_dim = self.backbone.feature_info[-1]['num_chs']
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.proj = nn.Linear(feat_dim, out_dim)
            self._use_timm = True
        else:
            # lightweight CNN fallback for demo / CI environments
            self._use_timm = False
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1))
            )
            self.proj = nn.Linear(64, out_dim)

    def forward(self, x):
        if self._use_timm:
            feats = self.backbone(x)[-1]  # B,C,H,W
            pooled = self.pool(feats).flatten(1)
            out = self.proj(pooled)
            return out
        else:
            feats = self.conv(x)
            pooled = feats.view(feats.size(0), -1)
            out = self.proj(pooled)
            return out
