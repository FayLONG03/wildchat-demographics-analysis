from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _normalize_topic_col(col: str) -> str | None:
    if not col.startswith("topic_"):
        return None
    suffix = col[len("topic_") :]
    return suffix if suffix.isdigit() else None


def build_merged_matrix(
    topic_proportions_path: Path,
    topic_labels_path: Path,
    output_parquet: Path,
    output_mapping_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    topic_df = pd.read_parquet(topic_proportions_path).copy()
    if "state" not in topic_df.columns:
        raise ValueError("Input topic proportions must contain a 'state' column.")

    label_df = pd.read_csv(topic_labels_path).copy()
    required = {"topic_id", "topic_label", "super_topic"}
    missing = required - set(label_df.columns)
    if missing:
        raise ValueError(f"Label file is missing required columns: {sorted(missing)}")

    label_df["topic_id"] = pd.to_numeric(label_df["topic_id"], errors="coerce")
    label_df = label_df.dropna(subset=["topic_id", "topic_label"]).copy()
    label_df["topic_id"] = label_df["topic_id"].astype(int)
    label_df["topic_col"] = label_df["topic_id"].map(lambda x: f"topic_{x}")

    raw_topic_cols = [c for c in topic_df.columns if _normalize_topic_col(c) is not None]
    available = set(raw_topic_cols)
    label_df = label_df[label_df["topic_col"].isin(available)].copy()
    if label_df.empty:
        raise ValueError("No overlap between label mapping and topic columns in input matrix.")

    grouped = (
        label_df.groupby("topic_label", as_index=False)
        .agg(
            super_topic=("super_topic", "first"),
            constituent_topics=("topic_col", lambda s: sorted(set(s))),
        )
        .sort_values("topic_label")
        .reset_index(drop=True)
    )
    grouped["merged_topic_id"] = [f"topic_merged_{i}" for i in range(len(grouped))]

    merged = topic_df[["state"]].copy()
    for _, row in grouped.iterrows():
        merged[row["merged_topic_id"]] = topic_df[row["constituent_topics"]].sum(axis=1)

    mapping = grouped[["merged_topic_id", "topic_label", "super_topic", "constituent_topics"]].copy()
    mapping["constituent_topics"] = mapping["constituent_topics"].map(lambda xs: "|".join(xs))
    mapping["n_constituent_topics"] = mapping["constituent_topics"].str.count(r"\|") + 1

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_parquet, index=False)
    mapping.to_csv(output_mapping_csv, index=False)
    return merged, mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build merged 71-topic state proportion matrix from raw BERTopic outputs."
    )
    parser.add_argument(
        "--topic-proportions",
        type=Path,
        default=Path("data/state_topic_proportions.parquet"),
        help="Raw state-topic proportions parquet (state + topic_0..topic_75).",
    )
    parser.add_argument(
        "--topic-labels",
        type=Path,
        default=Path("src/topic_modeling/results/topic_level_super_topics.csv"),
        help="Topic labeling file with topic_id/topic_label/super_topic.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/state_topic_proportions_merged71.parquet"),
        help="Output merged state-topic proportions parquet.",
    )
    parser.add_argument(
        "--output-mapping",
        type=Path,
        default=Path("results/statistics_merged71/merged_topic_mapping.csv"),
        help="Output merged topic mapping CSV for traceability.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merged_df, map_df = build_merged_matrix(
        topic_proportions_path=args.topic_proportions,
        topic_labels_path=args.topic_labels,
        output_parquet=args.output_parquet,
        output_mapping_csv=args.output_mapping,
    )
    print(
        f"Saved merged matrix: {args.output_parquet} "
        f"(rows={len(merged_df)}, merged_topics={len(merged_df.columns)-1})"
    )
    print(f"Saved mapping: {args.output_mapping} (rows={len(map_df)})")
