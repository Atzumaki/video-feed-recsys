from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class ModelConfig:
    num_items: int = 50_000
    num_categories: int = 200
    video_dim: int = 16
    dense_dim: int = 4
    d_model: int = 48
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class RankingBatch:
    history_item_ids: Tensor
    history_category_ids: Tensor
    candidate_item_ids: Tensor
    candidate_category_ids: Tensor
    dense_features: Tensor
    video_embeddings: Tensor


class TransformerRanker(nn.Module):
    """Sequence-aware multi-task ranker for watch and purchase probability."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        self.item_embedding = nn.Embedding(config.num_items + 1, config.d_model, padding_idx=0)
        self.category_embedding = nn.Embedding(
            config.num_categories + 1, config.d_model, padding_idx=0
        )
        self.position_embedding = nn.Embedding(50, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.video_projection = nn.Linear(config.video_dim, config.d_model)
        self.dense_projection = nn.Sequential(
            nn.Linear(config.dense_dim, config.d_model), nn.GELU(), nn.LayerNorm(config.d_model)
        )
        interaction_dim = config.d_model * 4
        self.shared = nn.Sequential(
            nn.Linear(interaction_dim, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
        )
        self.watch_head = nn.Linear(config.d_model, 1)
        self.purchase_head = nn.Linear(config.d_model, 1)

    def encode_history(self, item_ids: Tensor, category_ids: Tensor) -> Tensor:
        positions = torch.arange(item_ids.shape[1], device=item_ids.device)
        history = (
            self.item_embedding(item_ids)
            + self.category_embedding(category_ids)
            + self.position_embedding(positions)[None, :, :]
        )
        padding_mask = item_ids.eq(0)
        encoded = self.encoder(history, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1)
        return (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    def forward(self, batch: RankingBatch) -> dict[str, Tensor]:
        user = self.encode_history(batch.history_item_ids, batch.history_category_ids)
        candidate = (
            self.item_embedding(batch.candidate_item_ids)
            + self.category_embedding(batch.candidate_category_ids)
            + self.video_projection(batch.video_embeddings)
            + self.dense_projection(batch.dense_features)
        )
        expanded_user = user[:, None, :].expand_as(candidate)
        features = torch.cat(
            [
                expanded_user,
                candidate,
                expanded_user * candidate,
                torch.abs(expanded_user - candidate),
            ],
            dim=-1,
        )
        shared = self.shared(features)
        watch_logits = self.watch_head(shared).squeeze(-1)
        purchase_logits = self.purchase_head(shared).squeeze(-1)
        return {
            "watch_logits": watch_logits,
            "purchase_logits": purchase_logits,
            "ranking_score": 0.65 * watch_logits + 0.35 * purchase_logits,
        }

    @staticmethod
    def multitask_loss(
        outputs: dict[str, Tensor], watch_targets: Tensor, purchase_targets: Tensor
    ) -> Tensor:
        watch_loss = functional.binary_cross_entropy_with_logits(
            outputs["watch_logits"], watch_targets
        )
        purchase_loss = functional.binary_cross_entropy_with_logits(
            outputs["purchase_logits"], purchase_targets
        )
        return 0.65 * watch_loss + 0.35 * purchase_loss


class TwoTowerRetriever(nn.Module):
    """Normalized two-tower model suitable for ANN candidate retrieval."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.item_embedding = nn.Embedding(config.num_items + 1, config.d_model, padding_idx=0)
        self.category_embedding = nn.Embedding(
            config.num_categories + 1, config.d_model, padding_idx=0
        )
        self.video_projection = nn.Linear(config.video_dim, config.d_model)
        self.user_tower = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, 32)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, 32)
        )

    def encode_users(self, history_item_ids: Tensor) -> Tensor:
        mask = history_item_ids.ne(0).unsqueeze(-1)
        pooled = (self.item_embedding(history_item_ids) * mask).sum(1) / mask.sum(1).clamp_min(1)
        return functional.normalize(self.user_tower(pooled), dim=-1)

    def encode_items(self, item_ids: Tensor, category_ids: Tensor, video: Tensor) -> Tensor:
        features = (
            self.item_embedding(item_ids)
            + self.category_embedding(category_ids)
            + self.video_projection(video)
        )
        return functional.normalize(self.item_tower(features), dim=-1)

    def forward(
        self, history_item_ids: Tensor, item_ids: Tensor, category_ids: Tensor, video: Tensor
    ) -> Tensor:
        return self.encode_users(history_item_ids) @ self.encode_items(
            item_ids, category_ids, video
        ).T

