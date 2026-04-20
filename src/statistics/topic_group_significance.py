from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import benjamini_hochberg, ensure_output_dir, get_topic_columns


def weighted_dispersion(topic_values: np.ndarray, weights: np.ndarray) -> float:
    weighted_mean = np.average(topic_values, weights=weights)
    return float(np.sum(weights * (topic_values - weighted_mean) ** 2) / np.sum(weights))


def heterogeneity_p_value(
    topic_values: np.ndarray,
    state_doc_counts: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    observed = weighted_dispersion(topic_values, state_doc_counts)
    global_p = float(np.average(topic_values, weights=state_doc_counts))
    sim_stats = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        sim_topic_props = rng.binomial(state_doc_counts, global_p) / state_doc_counts
        sim_stats[i] = weighted_dispersion(sim_topic_props, state_doc_counts)
    # Add-one smoothing avoids reporting exact zero with finite permutations.
    p_value = float(((sim_stats >= observed).sum() + 1) / (n_perm + 1))
    return observed, p_value


def run_significance_tests(
    topic_proportions_path: Path,
    state_docs_input_path: Path,
    output_dir: Path,
    n_perm: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    ensure_output_dir(output_dir)
    rng = np.random.default_rng(seed)

    topic_df = pd.read_parquet(topic_proportions_path).copy()
    topic_df["state"] = topic_df["state"].astype(str).str.strip()

    state_docs_df = pd.read_parquet(state_docs_input_path).copy()
    state_docs_df["state"] = state_docs_df["state"].astype(str).str.strip()
    state_doc_counts = (
        state_docs_df[state_docs_df["state"].str.len() > 0]["state"]
        .value_counts()
        .rename_axis("state")
        .reset_index(name="state_docs_n")
    )
    merged = topic_df.merge(state_doc_counts, on="state", how="inner")

    topic_cols = get_topic_columns(merged.columns)
    n_states = merged["state"].nunique()
    if n_states < 5:
        raise ValueError(f"Need at least 5 states with documents. Found {n_states}.")

    all_results = []
    weights = merged["state_docs_n"].to_numpy().astype(int)
    for topic in topic_cols:
        values = merged[topic].to_numpy()
        observed, p_perm = heterogeneity_p_value(
            topic_values=values,
            state_doc_counts=weights,
            n_perm=n_perm,
            rng=rng,
        )
        all_results.append(
            {
                "topic": topic,
                "weighted_dispersion": observed,
                "perm_p_value": p_perm,
                "n_states": n_states,
                "total_docs": int(weights.sum()),
            }
        )

    results = pd.DataFrame(all_results)
    results["fdr_q_value"] = benjamini_hochberg(results["perm_p_value"])
    results["significant_fdr"] = results["fdr_q_value"] < alpha
    results = results.sort_values(["fdr_q_value", "perm_p_value"], ascending=True)

    results.to_csv(output_dir / "topic_state_heterogeneity_significance.csv", index=False)
    results.loc[results["significant_fdr"]].to_csv(
        output_dir / "topic_state_heterogeneity_significant.csv", index=False
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="State-only topic heterogeneity significance tests."
    )
    parser.add_argument(
        "--topic-proportions",
        type=Path,
        default=Path("data/state_topic_proportions.parquet"),
        help="Path to state-level topic proportion parquet.",
    )
    parser.add_argument(
        "--state-docs-input",
        type=Path,
        default=Path("data/wildchat_bertopic.parquet"),
        help="Path with state-labeled user turns to derive per-state document counts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/statistics"),
        help="Directory to store significance outputs.",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=5000,
        help="Number of permutations per topic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha for FDR and bootstrap interval.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = run_significance_tests(
        topic_proportions_path=args.topic_proportions,
        state_docs_input_path=args.state_docs_input,
        output_dir=args.output_dir,
        n_perm=args.n_perm,
        seed=args.seed,
        alpha=args.alpha,
    )
    print(f"Saved {len(output)} topic significance rows to {args.output_dir}")
