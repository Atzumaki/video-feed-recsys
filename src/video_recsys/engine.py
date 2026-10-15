from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from video_recsys.model import ModelConfig, RankingBatch, TransformerRanker
from video_recsys.reranking import business_score, diversify
from video_recsys.schemas import RankedItem, RankRequest, RankResponse


class RankingEngine:
    def __init__(self, artifact_path: Path | None = None) -> None:
        torch.manual_seed(2026)
        self.config = ModelConfig()
        self.model = TransformerRanker(self.config)
        self.model_version = "transformer-mtl-demo-v1"
        if artifact_path and artifact_path.exists():
            checkpoint = torch.load(artifact_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(checkpoint)
            self.model_version = artifact_path.stem
        self.model.eval()

    def rank(self, request: RankRequest) -> RankResponse:
        history_items = request.recent_item_ids[-50:] or [0]
        history_categories = request.recent_category_ids[-50:] or [0]
        candidates = request.candidates
        dense = np.array(
            [
                [
                    item.retrieval_score,
                    item.quality_score,
                    np.log1p(item.freshness_hours) / 8,
                    np.log1p(item.ad_cpm) / 6,
                ]
                for item in candidates
            ],
            dtype=np.float32,
        )
        video = np.asarray([item.video_embedding for item in candidates], dtype=np.float32)
        batch = RankingBatch(
            history_item_ids=torch.tensor([history_items]),
            history_category_ids=torch.tensor([history_categories]),
            candidate_item_ids=torch.tensor([[item.item_id for item in candidates]]),
            candidate_category_ids=torch.tensor([[item.category_id for item in candidates]]),
            dense_features=torch.from_numpy(dense)[None, :, :],
            video_embeddings=torch.from_numpy(video)[None, :, :],
        )
        with torch.inference_mode():
            outputs = self.model(batch)
            watch = torch.sigmoid(outputs["watch_logits"])[0].numpy()
            purchase = torch.sigmoid(outputs["purchase_logits"])[0].numpy()
        utility = business_score(
            watch, purchase, np.asarray([item.ad_cpm for item in candidates], dtype=float)
        )
        selected = diversify(
            utility,
            video,
            np.asarray([item.category_id for item in candidates]),
            top_k=min(request.top_k, len(candidates)),
            diversity_lambda=request.diversity_lambda,
        )
        items = [
            RankedItem(
                item_id=candidates[index].item_id,
                category_id=candidates[index].category_id,
                rank=rank,
                score=round(float(utility[index]), 6),
                watch_probability=round(float(watch[index]), 6),
                purchase_probability=round(float(purchase[index]), 6),
            )
            for rank, index in enumerate(selected, start=1)
        ]
        return RankResponse(user_id=request.user_id, model_version=self.model_version, items=items)

