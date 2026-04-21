from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import ensure_output_dir, get_topic_columns


def run_state_topic_analysis(
    topic_proportions_path: Path,
    output_dir: Path,
    top_k_per_topic: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_dir(output_dir)

    topic_df = pd.read_parquet(topic_proportions_path).copy()
    topic_df["state"] = topic_df["state"].astype(str).str.strip()
    topic_cols = get_topic_columns(topic_df.columns)

    long_df = topic_df.melt(
        id_vars=["state"],
        value_vars=topic_cols,
        var_name="topic",
        value_name="topic_proportion",
    )

    # Standardize within each topic so we can compare over-/under-indexing by state.
    long_df["topic_zscore"] = long_df.groupby("topic")["topic_proportion"].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12)
    )
    long_df["rank_within_topic_desc"] = long_df.groupby("topic")["topic_proportion"].rank(
        method="min",
        ascending=False,
    )

    variation_rows = []
    for topic in topic_cols:
        values = topic_df[topic].to_numpy()
        mean_val = float(values.mean())
        std_val = float(values.std(ddof=0))
        max_state_idx = int(np.argmax(values))
        min_state_idx = int(np.argmin(values))
        variation_rows.append(
            {
                "topic": topic,
                "mean_proportion": mean_val,
                "std_proportion": std_val,
                "cv_proportion": std_val / (mean_val + 1e-12),
                "max_state": topic_df.iloc[max_state_idx]["state"],
                "max_proportion": float(values[max_state_idx]),
                "min_state": topic_df.iloc[min_state_idx]["state"],
                "min_proportion": float(values[min_state_idx]),
                "range_proportion": float(values[max_state_idx] - values[min_state_idx]),
            }
        )
    variation_df = pd.DataFrame(variation_rows).sort_values(
        "std_proportion",
        ascending=False,
    )

    top_over = long_df.loc[long_df["rank_within_topic_desc"] <= top_k_per_topic].copy()
    top_under = long_df.sort_values(["topic", "topic_proportion"], ascending=[True, True]).groupby(
        "topic", as_index=False, group_keys=False
    ).head(top_k_per_topic)

    long_df.to_csv(output_dir / "state_topic_long.csv", index=False)
    variation_df.to_csv(output_dir / "topic_variation_by_state.csv", index=False)
    top_over.to_csv(output_dir / f"state_topic_top_{top_k_per_topic}_states.csv", index=False)
    top_under.to_csv(output_dir / f"state_topic_bottom_{top_k_per_topic}_states.csv", index=False)
    return long_df, variation_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run state-only topic concentration and variation analysis."
    )
    parser.add_argument(
        "--topic-proportions",
        type=Path,
        default=Path("data/state_topic_proportions.parquet"),
        help="Path to state-level topic proportion parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/statistics"),
        help="Directory to store state-topic analysis outputs.",
    )
    parser.add_argument(
        "--top-k-per-topic",
        type=int,
        default=5,
        help="How many highest/lowest states to export per topic.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    state_topic_long, topic_variation = run_state_topic_analysis(
        topic_proportions_path=args.topic_proportions,
        output_dir=args.output_dir,
        top_k_per_topic=args.top_k_per_topic,
    )
    print(
        f"Saved {len(state_topic_long)} state-topic rows and "
        f"{len(topic_variation)} topic-variation rows to {args.output_dir}"
    )
