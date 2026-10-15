import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from video_recsys import __version__
from video_recsys.engine import RankingEngine
from video_recsys.schemas import RankRequest, RankResponse

app = FastAPI(
    title="Video Feed Recommender",
    version=__version__,
    description="Two-stage neural ranking API for a personalized commerce video feed.",
)
artifact = os.getenv("MODEL_ARTIFACT")
engine = RankingEngine(Path(artifact) if artifact else None)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "model": engine.model_version}


@app.post("/v1/rank", response_model=RankResponse)
def rank(request: RankRequest) -> RankResponse:
    try:
        return engine.rank(request)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

