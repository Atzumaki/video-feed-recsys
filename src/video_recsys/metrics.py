from __future__ import annotations

import numpy as np


def ndcg_at_k(relevance: list[float], scores: list[float], k: int) -> float:
    relevance_array = np.asarray(relevance, dtype=float)
    order = np.argsort(-np.asarray(scores, dtype=float))[:k]
    ideal = np.argsort(-relevance_array)[:k]

    def dcg(indices: np.ndarray) -> float:
        gains = np.power(2, relevance_array[indices]) - 1
        discounts = np.log2(np.arange(len(indices)) + 2)
        return float(np.sum(gains / discounts))

    ideal_dcg = dcg(ideal)
    return dcg(order) / ideal_dcg if ideal_dcg else 0.0


def recall_at_k(relevant_items: set[int], ranked_items: list[int], k: int) -> float:
    if not relevant_items:
        return 0.0
    return len(relevant_items.intersection(ranked_items[:k])) / len(relevant_items)


def category_coverage(categories: list[int], catalog_categories: set[int], k: int) -> float:
    if not catalog_categories:
        return 0.0
    return len(set(categories[:k])) / len(catalog_categories)


def intra_list_diversity(embeddings: np.ndarray) -> float:
    if len(embeddings) < 2:
        return 0.0
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    similarity = normalized @ normalized.T
    upper = similarity[np.triu_indices(len(embeddings), k=1)]
    return float(np.mean(1 - upper))

