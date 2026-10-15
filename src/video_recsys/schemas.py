from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Candidate(BaseModel):
    item_id: int = Field(ge=1, le=50_000)
    category_id: int = Field(ge=1, le=200)
    retrieval_score: float = Field(ge=-1, le=1)
    quality_score: float = Field(ge=0, le=1)
    freshness_hours: float = Field(ge=0)
    ad_cpm: float = Field(default=0, ge=0)
    video_embedding: list[float] = Field(min_length=16, max_length=16)


class RankRequest(BaseModel):
    user_id: int = Field(ge=1)
    recent_item_ids: list[int] = Field(default_factory=list, max_length=50)
    recent_category_ids: list[int] = Field(default_factory=list, max_length=50)
    candidates: list[Candidate] = Field(min_length=1, max_length=200)
    top_k: int = Field(default=20, ge=1, le=50)
    diversity_lambda: float = Field(default=0.18, ge=0, le=1)

    @model_validator(mode="after")
    def validate_history(self) -> RankRequest:
        if len(self.recent_item_ids) != len(self.recent_category_ids):
            raise ValueError("recent item and category histories must have equal length")
        if any(item < 1 or item > 50_000 for item in self.recent_item_ids):
            raise ValueError("history item ids must be in [1, 50000]")
        if any(category < 1 or category > 200 for category in self.recent_category_ids):
            raise ValueError("history category ids must be in [1, 200]")
        return self


class RankedItem(BaseModel):
    item_id: int
    category_id: int
    rank: int
    score: float
    watch_probability: float
    purchase_probability: float


class RankResponse(BaseModel):
    user_id: int
    model_version: str
    items: list[RankedItem]


class ABTestResult(BaseModel):
    control_mean: float
    treatment_mean: float
    absolute_uplift: float
    relative_uplift_percent: float
    confidence_interval_95: tuple[float, float]
    p_value: float
    significant: bool

