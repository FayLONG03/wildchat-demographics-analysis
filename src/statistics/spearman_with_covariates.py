from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from utils import benjamini_hochberg, ensure_output_dir, get_topic_columns


def run_spearman(
    topic_proportions_path: Path,
    covariates_path: Path,
    output_dir: Path,
    alpha: float = 0.05,
) -> pd.DataFrame:
    ensure_output_dir(output_dir)

    topic_df = pd.read_parquet(topic_proportions_path).copy()
    topic_df["state"] = topic_df["state"].astype(str).str.strip()

    cov_df = pd.read_csv(covariates_path).copy()
    cov_df["state"] = cov_df["state"].astype(str).str.strip()
    cov_df["income"] = pd.to_numeric(cov_df["income"], errors="coerce")
    cov_df["education"] = pd.to_numeric(cov_df["education"], errors="coerce")
    cov_df = cov_df.dropna(subset=["state", "income", "education"]).drop_duplicates(subset=["state"])

    merged = topic_df.merge(cov_df, on="state", how="inner")
    if merged.empty:
        raise ValueError("Merged table is empty. Check state names and covariate files.")

    topic_cols = get_topic_columns(merged.columns)
    rows: list[dict[str, object]] = []
    for covariate in ["income", "education"]:
        for topic in topic_cols:
            rho, p_value = spearmanr(merged[topic], merged[covariate], nan_policy="omit")
            rows.append(
                {
                    "covariate": covariate,
                    "topic": topic,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "n_states": merged[["state", topic, covariate]].dropna().shape[0],
                }
            )

    out = pd.DataFrame(rows)
    out["fdr_q_value"] = out.groupby("covariate", group_keys=False)["p_value"].apply(benjamini_hochberg)
    out["significant_fdr"] = out["fdr_q_value"] < alpha
    out = out.sort_values(["covariate", "fdr_q_value", "p_value"], ascending=[True, True, True])

    merged.to_csv(output_dir / "state_topic_with_covariates.csv", index=False)
    out.to_csv(output_dir / "spearman_topic_covariates.csv", index=False)
    out.loc[out["significant_fdr"]].to_csv(
        output_dir / "spearman_topic_covariates_significant.csv",
        index=False,
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Spearman correlations between topic proportions and state covariates."
    )
    parser.add_argument(
        "--topic-proportions",
        type=Path,
        default=Path("data/state_topic_proportions.parquet"),
        help="Path to state topic proportion parquet.",
    )
    parser.add_argument(
        "--covariates",
        type=Path,
        default=Path("data/state_covariates.csv"),
        help="Path to cleaned state covariates CSV (state, income, education).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/statistics"),
        help="Directory for Spearman output files.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR significance threshold.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_spearman(
        topic_proportions_path=args.topic_proportions,
        covariates_path=args.covariates,
        output_dir=args.output_dir,
        alpha=args.alpha,
    )
    print(f"Saved {len(result)} Spearman rows to {args.output_dir}")
