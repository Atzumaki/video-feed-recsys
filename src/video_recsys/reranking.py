from __future__ import annotations

import numpy as np


def business_score(
    watch_probability: np.ndarray,
    purchase_probability: np.ndarray,
    ad_cpm: np.ndarray,
) -> np.ndarray:
    normalized_cpm = ad_cpm / max(float(ad_cpm.max()), 1.0)
    return 0.58 * watch_probability + 0.34 * purchase_probability + 0.08 * normalized_cpm


def diversify(
    scores: np.ndarray,
    embeddings: np.ndarray,
    categories: np.ndarray,
    *,
    top_k: int,
    diversity_lambda: float,
) -> list[int]:
    """Greedy maximum-marginal-relevance reranking with category fatigue."""
    if len(scores) == 0:
        return []
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    similarity = normalized @ normalized.T
    chosen: list[int] = []
    remaining = set(range(len(scores)))
    category_counts: dict[int, int] = {}
    while remaining and len(chosen) < min(top_k, len(scores)):
        best_index = max(
            remaining,
            key=lambda index: float(scores[index])
            - diversity_lambda
            * (
                max((float(similarity[index, previous]) for previous in chosen), default=0)
                + 0.25 * category_counts.get(int(categories[index]), 0)
            ),
        )
        chosen.append(best_index)
        remaining.remove(best_index)
        category = int(categories[best_index])
        category_counts[category] = category_counts.get(category, 0) + 1
    return chosen

