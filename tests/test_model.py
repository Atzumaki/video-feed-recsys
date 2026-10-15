import torch

from video_recsys.model import ModelConfig, TransformerRanker, TwoTowerRetriever
from video_recsys.synthetic import synthetic_batch


def test_ranker_shapes_and_backward() -> None:
    config = ModelConfig(num_items=500, num_categories=20, d_model=24, num_heads=4, num_layers=1)
    model = TransformerRanker(config)
    batch, watch, purchase = synthetic_batch(
        batch_size=3, candidate_count=5, config=config
    )
    outputs = model(batch)
    assert outputs["ranking_score"].shape == (3, 5)
    loss = model.multitask_loss(outputs, watch, purchase)
    loss.backward()
    assert model.item_embedding.weight.grad is not None


def test_two_tower_embeddings_are_normalized() -> None:
    config = ModelConfig(num_items=100, num_categories=10, d_model=24)
    model = TwoTowerRetriever(config)
    encoded = model.encode_items(
        torch.tensor([1, 2]),
        torch.tensor([1, 2]),
        torch.randn(2, config.video_dim),
    )
    assert torch.allclose(torch.linalg.vector_norm(encoded, dim=1), torch.ones(2), atol=1e-5)

