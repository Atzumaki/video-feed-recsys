import argparse
import json
from pathlib import Path

import numpy as np

from video_recsys.engine import RankingEngine
from video_recsys.experiments import analyze_ab_test
from video_recsys.schemas import RankRequest
from video_recsys.synthetic import synthetic_candidates
from video_recsys.training import train_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end video recommender demo")
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    losses = train_demo(steps=args.train_steps, output=args.artifact)
    request = RankRequest(
        user_id=42,
        recent_item_ids=[12, 98, 451, 87, 22],
        recent_category_ids=[2, 4, 2, 7, 2],
        candidates=synthetic_candidates(),
        top_k=10,
    )
    ranking = RankingEngine(args.artifact).rank(request)
    random = np.random.default_rng(42)
    pre = random.normal(12, 3, size=2000)
    control = 0.65 * pre + random.normal(4, 1.8, size=2000)
    treatment = 0.65 * pre + random.normal(4.25, 1.8, size=2000)
    experiment = analyze_ab_test(control, treatment, control_pre=pre, treatment_pre=pre)
    print(
        json.dumps(
            {
                "training_loss": {"first": losses[0], "last": losses[-1]},
                "top_items": [item.model_dump() for item in ranking.items],
                "ab_test": experiment.model_dump(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

