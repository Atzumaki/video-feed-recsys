import numpy as np

from video_recsys.reranking import diversify


def test_diversifier_returns_unique_items_and_breaks_similarity_cluster() -> None:
    scores = np.array([1.0, 0.99, 0.8])
    embeddings = np.array([[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]])
    categories = np.array([1, 1, 2])
    result = diversify(
        scores, embeddings, categories, top_k=2, diversity_lambda=0.5
    )
    assert result == [0, 2]
    assert len(result) == len(set(result))

