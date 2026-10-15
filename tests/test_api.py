from fastapi.testclient import TestClient

from video_recsys.api import app
from video_recsys.synthetic import synthetic_candidates

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rank_endpoint() -> None:
    response = client.post(
        "/v1/rank",
        json={
            "user_id": 42,
            "recent_item_ids": [1, 2, 3],
            "recent_category_ids": [1, 2, 1],
            "candidates": synthetic_candidates(8),
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 5
    assert [item["rank"] for item in body["items"]] == [1, 2, 3, 4, 5]

