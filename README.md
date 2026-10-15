# Video Feed Recommender

Production-oriented pet project for a personalized short-video commerce feed.
It demonstrates the complete DS loop: point-in-time data preparation, candidate
retrieval, sequence-aware neural ranking, business-aware diversification,
online serving, offline evaluation and A/B-test analysis.

> The project is not affiliated with Ozon. All demo data is synthetic.

## Why this project is relevant

The repository mirrors the responsibilities of a full-cycle recommendation DS:

- **candidate retrieval** — a two-tower PyTorch model with normalized vectors,
  ready for an ANN index;
- **neural ranking** — a Transformer encodes recent behavior and a multi-task
  head predicts watch and purchase probabilities;
- **multimodality** — video embeddings enter both retrieval and ranking stages;
- **product constraints** — the final utility combines engagement, purchase
  intent and ad value, then applies maximum-marginal-relevance diversification;
- **experimentation** — NDCG, Recall, coverage, intra-list diversity and a
  CUPED-enabled A/B-test analyzer;
- **data platform** — point-in-time ClickHouse SQL, an example PySpark feature
  job and an Airflow DAG;
- **production path** — typed FastAPI contract, artifact loading, Docker and CI.

## Architecture

```text
events ──► Spark / SQL features ──► training set
                                         │
catalog + video embeddings ──► two-tower retrieval ──► ANN candidates
                                                          │
recent user sequence ──► Transformer ranker ──► P(watch), P(purchase)
                                                          │
                              utility + MMR diversification ──► feed
                                                          │
                         offline metrics + CUPED A/B analysis
```

## Model

The ranker encodes up to 50 recent interactions with a Transformer. Candidate
item, category, dense and video features interact with the user vector through
concatenation, element-wise product and absolute distance. Two heads optimize:

```text
L = 0.65 · BCE(watch) + 0.35 · BCE(purchase)
```

The final online utility is intentionally separate from learned logits so that
product weights can be audited and changed without retraining:

```text
utility = 0.58 · P(watch) + 0.34 · P(purchase) + 0.08 · normalized_ad_CPM
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
video-recsys --train-steps 10 --artifact artifacts/demo.pt
```

Run the API:

```bash
uvicorn video_recsys.api:app --reload
```

Swagger UI: `http://127.0.0.1:8000/docs`; ranking endpoint: `POST /v1/rank`.

## Quality gates

```bash
ruff check .
pytest --cov=video_recsys
```

Tests cover gradient flow and output shapes, normalized retrieval embeddings,
MMR behavior, ranking metrics, CUPED variance reduction, uplift detection and
the HTTP contract.

## Important engineering decisions

1. **No target leakage:** SQL and Spark windows end before the current impression.
2. **Multi-task targets:** watch quality and shopping intent remain observable
   separately instead of being collapsed into one label.
3. **Business policy outside the network:** revenue weighting and diversity are
   explicit, testable post-processing stages.
4. **Train/serve parity:** one typed `RankingBatch` is used by training and serving.
5. **Honest scope:** the checked-in model is trained on synthetic data; the project
   demonstrates architecture and engineering, not production metric claims.

## Next experiments

- sampled-softmax retrieval training and FAISS/HNSW serving;
- causal debiasing for position and selection bias;
- PyTorch Lightning + Hydra experiment configuration;
- Triton ensemble for retrieval and ranking;
- sequential counterfactual evaluation before online rollout;
- Go gateway with request coalescing and strict latency budgets.

