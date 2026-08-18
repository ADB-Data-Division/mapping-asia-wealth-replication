from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

from torchgeo.models import swin_v2_t
from torchgeo.models.swin import Swin_V2_T_Weights


class CrossAttention(nn.Module):
    """Multi-head cross-attention using scaled dot-product attention."""

    def __init__(self, dim: int, num_heads: int, out_proj_scale: float = 1.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self._init_weights(out_proj_scale)

    def _init_weights(self, out_proj_scale: float):
        # Xavier uniform for Q, K, V projections
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        # Output projection scaled down for residual stability
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        with torch.no_grad():
            self.out_proj.weight.mul_(out_proj_scale)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q:  [B, Nq, C] query tokens
            kv: [B, Nk, C] key-value tokens
        Returns:
            [B, Nq, C] attended output
        """
        B, Nq, C = q.shape
        Nk = kv.size(1)

        q = self.q_proj(q).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, Nq, C)
        return self.out_proj(out)


class TokenConditioningBlock(nn.Module):
    """
    Fuses image features and spatial covariates into a query token.
      1. Global average pool stage features -> image token
      2. Project image token to token_dim (if stage_dim != token_dim)
      3. Project covariate embedding (128-dim) to token_dim via per-block cov_encoder
      4. Concatenate [image_token, covariate_token] as key-value
      5. Cross-attention: query attends to key-value
      6. FFN with residual connections and stochastic depth
    """

    def __init__(
        self,
        token_dim: int,
        stage_dim: int,
        num_heads: int,
        cov_dim: int = 128,
        drop_path_rate: float = 0.3,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.stage_dim = stage_dim

        # Per-block covariate projection
        self.cov_encoder = nn.Sequential(
            nn.Linear(cov_dim, token_dim, bias=True),
            nn.LayerNorm(token_dim),
        )

        # Project stage features to token_dim if dimensions differ
        self.stage_proj = nn.Linear(stage_dim, token_dim) if stage_dim != token_dim else nn.Identity()

        self.cross_attn = CrossAttention(dim=token_dim, num_heads=num_heads, out_proj_scale=residual_scale)
        self.norm1 = nn.LayerNorm(token_dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(token_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(token_dim, eps=1e-6)
        self.drop_path = StochasticDepth(drop_path_rate, mode="row")

        self.mlp_fc1 = nn.Linear(token_dim, token_dim * 4)
        self.mlp_fc2 = nn.Linear(token_dim * 4, token_dim)
        self._init_mlp_weights(residual_scale)

    def _init_mlp_weights(self, residual_scale: float):
        nn.init.xavier_uniform_(self.mlp_fc1.weight)
        nn.init.zeros_(self.mlp_fc1.bias)
        nn.init.xavier_uniform_(self.mlp_fc2.weight)
        nn.init.zeros_(self.mlp_fc2.bias)
        # Scale down output projection for residual stability
        with torch.no_grad():
            self.mlp_fc2.weight.mul_(residual_scale)

    def forward(
        self,
        query: torch.Tensor,
        stage_feat: torch.Tensor,
        cov_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query:      [B, 1, token_dim] learned query token
            stage_feat: [B, C, H, W] feature map from current Swin stage
            cov_emb:    [B, 1, 128] embedded spatial covariates
        Returns:
            [B, 1, token_dim] updated query token
        """
        # Image token: spatial mean pooling -> [B, 1, stage_dim] -> [B, 1, token_dim]
        img_token = stage_feat.flatten(2).mean(dim=2, keepdim=True).transpose(1, 2)
        img_token = self.stage_proj(img_token)

        # Covariate token: project 128-dim embedding to token_dim
        cov_token = self.cov_encoder(cov_emb)

        # Key-value: concatenate image and covariate tokens [B, 2, token_dim]
        kv = torch.cat([img_token, cov_token], dim=1)

        # Cross-attention with pre-norm and residual
        query = query + self.drop_path(self.cross_attn(self.norm1(query), self.norm_kv(kv)))
        query = query + self.drop_path(self.mlp_fc2(F.gelu(self.mlp_fc1(self.norm2(query)))))
        return query


class SwinV2TBackbone(nn.Module):
    """
    SwinV2-T backbone with stage-wise access.

    Each stage outputs NHWC format which we convert to NCHW for downstream use.
    """

    STAGE_CHANNELS = [96, 192, 384, 768]

    def __init__(self, weights: Optional[Swin_V2_T_Weights] = None):
        super().__init__()
        swin = swin_v2_t(weights=weights)

        self.stages = nn.ModuleList([
            swin.features[:2],
            swin.features[2:4],
            swin.features[4:6],
            swin.features[6:],
        ])

    @property
    def token_dim(self) -> int:
        return self.STAGE_CHANNELS[-1]


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 9
    cov_channels: int = 29
    drop_path_rate: float = 0.3
    head_dropout: float = 0.3


class MultimodalSwinV2TRegressor(nn.Module):
    """
    Multimodal regression model combining satellite imagery and spatial covariates.

    Forward pass:
      1. Embed spatial covariates: [B, cov_channels] -> [B, 1, 128]
      2. Initialize learned query token: [B, 1, 768]
      3. For each Swin stage:
         - Extract stage features
         - Update query via TokenConditioningBlock (cross-attention to image + covariates)
      4. Regression head on final query token -> scalar prediction
    """

    # Number of attention heads per stage (matches original for swin_v2_t)
    HEADS_PER_STAGE = [3, 6, 12, 24]

    def __init__(
        self,
        cfg: ModelConfig = ModelConfig(),
        weights: Optional[Swin_V2_T_Weights] = Swin_V2_T_Weights.SENTINEL2_SI_MS_SATLAS,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.cfg = cfg
        self.freeze_backbone = freeze_backbone

        self.backbone = SwinV2TBackbone(weights=weights)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        token_dim = self.backbone.token_dim

        # Shared covariate embedding: [B, cov_channels] -> [B, 1, 128]
        self.cov_embedding = nn.Sequential(
            nn.Linear(cfg.cov_channels, 64, bias=True),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128, bias=True),
            nn.LayerNorm(128),
        )

        # Per-stage conditioning blocks with scaled residuals
        num_residual_blocks = 4
        residual_scale = (2 * num_residual_blocks) ** -0.5
        self.conditioners = nn.ModuleList([
            TokenConditioningBlock(
                token_dim=token_dim,
                stage_dim=self.backbone.STAGE_CHANNELS[i],
                num_heads=self.HEADS_PER_STAGE[i],
                cov_dim=128,
                drop_path_rate=cfg.drop_path_rate,
                residual_scale=residual_scale,
            )
            for i in range(4)
        ])

        # Learned query token
        self.query_token = nn.Embedding(1, token_dim)
        nn.init.trunc_normal_(self.query_token.weight, std=0.02)

        # Regression head
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(token_dim, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()            # prevent stochastic depth in frozen Swin
        return self

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 9, H, W] Sentinel-2 imagery (9 channels)
            g: [B, cov_channels] spatial covariates
        Returns:
            [B] scalar predictions
        """
        B = x.size(0)

        # Embed covariates
        cov_emb = self.cov_embedding(g.unsqueeze(1) if g.dim() == 2 else g)  # [B, 1, 128]

        # Initialize query token for batch
        query = self.query_token.weight.unsqueeze(0).expand(B, -1, -1)  # [B, 1, 768]

        # Process through Swin stages with conditioning
        for i, stage in enumerate(self.backbone.stages):
            if self.freeze_backbone:
                with torch.no_grad():
                    x = stage(x)                           # [B, H, W, C] (NHWC)
            else:
                x = stage(x)
            x_nchw = x.permute(0, 3, 1, 2).contiguous()    # [B, C, H, W]
            query = self.conditioners[i](query, x_nchw, cov_emb)

        # Regression head
        out = self.head(query)  # [B, 1, 1]
        return out.squeeze(-1).squeeze(-1)  # [B]

