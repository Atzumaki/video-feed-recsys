import numpy as np
import pytest

from video_recsys.metrics import intra_list_diversity, ndcg_at_k, recall_at_k


def test_perfect_ranking_has_unit_ndcg() -> None:
    assert ndcg_at_k([3, 2, 0], [0.9, 0.5, 0.1], 3) == pytest.approx(1.0)


def test_recall_and_diversity() -> None:
    assert recall_at_k({2, 4}, [1, 2, 3, 4], 3) == 0.5
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert intra_list_diversity(embeddings) == pytest.approx(1.0)

