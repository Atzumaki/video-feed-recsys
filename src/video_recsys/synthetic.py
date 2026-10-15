from __future__ import annotations

import numpy as np
import torch

from video_recsys.model import ModelConfig, RankingBatch


def synthetic_batch(
    batch_size: int = 16,
    history_length: int = 20,
    candidate_count: int = 12,
    *,
    config: ModelConfig | None = None,
    seed: int = 42,
) -> tuple[RankingBatch, torch.Tensor, torch.Tensor]:
    config = config or ModelConfig()
    generator = torch.Generator().manual_seed(seed)
    history_items = torch.randint(
        1, config.num_items + 1, (batch_size, history_length), generator=generator
    )
    history_categories = torch.randint(
        1, config.num_categories + 1, (batch_size, history_length), generator=generator
    )
    candidate_items = torch.randint(
        1, config.num_items + 1, (batch_size, candidate_count), generator=generator
    )
    candidate_categories = torch.randint(
        1, config.num_categories + 1, (batch_size, candidate_count), generator=generator
    )
    dense = torch.rand(batch_size, candidate_count, config.dense_dim, generator=generator)
    video = torch.randn(batch_size, candidate_count, config.video_dim, generator=generator)
    affinity = history_categories[:, -5:, None].eq(candidate_categories[:, None, :]).float().sum(1)
    watch_probability = torch.sigmoid(-1.1 + 0.75 * affinity + 1.2 * dense[..., 0])
    purchase_probability = torch.sigmoid(-3.3 + 0.55 * affinity + 1.5 * dense[..., 1])
    watch = torch.bernoulli(watch_probability, generator=generator)
    purchase = torch.bernoulli(purchase_probability, generator=generator)
    batch = RankingBatch(
        history_item_ids=history_items,
        history_category_ids=history_categories,
        candidate_item_ids=candidate_items,
        candidate_category_ids=candidate_categories,
        dense_features=dense,
        video_embeddings=video,
    )
    return batch, watch, purchase


def synthetic_candidates(count: int = 30, seed: int = 7) -> list[dict[str, object]]:
    random = np.random.default_rng(seed)
    result = []
    for index in range(count):
        result.append(
            {
                "item_id": index + 101,
                "category_id": int(random.integers(1, 9)),
                "retrieval_score": float(random.uniform(0.1, 0.95)),
                "quality_score": float(random.uniform(0.3, 1)),
                "freshness_hours": float(random.uniform(1, 240)),
                "ad_cpm": float(random.choice([0, 0, 0, random.uniform(20, 180)])),
                "video_embedding": random.normal(size=16).tolist(),
            }
        )
    return result

