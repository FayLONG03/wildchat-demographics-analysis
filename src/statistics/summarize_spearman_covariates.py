from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def _topic_words_map(representative_prompts_path: Path) -> dict[str, str]:
    rep = pd.read_csv(representative_prompts_path)
    rep = rep[["topic_id", "top_words"]].drop_duplicates("topic_id")
    rep["topic"] = rep["topic_id"].apply(lambda x: f"topic_{int(x)}")
    return dict(zip(rep["topic"], rep["top_words"]))


def build_experiment2_snapshot(
    state_topic_with_covariates_path: Path,
    spearman_path: Path,
    spearman_sig_path: Path,
    representative_prompts_path: Path,
    output_path: Path,
) -> str:
    merged = pd.read_csv(state_topic_with_covariates_path)
    spearman = pd.read_csv(spearman_path)
    sig = pd.read_csv(spearman_sig_path)
    topic_words = _topic_words_map(representative_prompts_path)

    n_states = merged["state"].nunique()
    n_topics = len([c for c in merged.columns if c.startswith("topic_")])
    n_tests = len(spearman)
    sig_income = int((sig["covariate"] == "income").sum())
    sig_education = int((sig["covariate"] == "education").sum())

    income_edu_rho, income_edu_p = spearmanr(
        merged["income"],
        merged["education"],
        nan_policy="omit",
    )

    def top_abs_rho_rows(covariate: str, k: int = 5) -> pd.DataFrame:
        temp = spearman[spearman["covariate"] == covariate].copy()
        temp["abs_rho"] = temp["spearman_rho"].abs()
        temp = temp.sort_values(["abs_rho", "p_value"], ascending=[False, True]).head(k)
        temp["top_words"] = temp["topic"].map(topic_words).fillna("")
        return temp

    top_income = top_abs_rho_rows("income", 5)
    top_education = top_abs_rho_rows("education", 5)

    sig_with_words = sig.copy()
    sig_with_words["top_words"] = sig_with_words["topic"].map(topic_words).fillna("")
    sig_with_words = sig_with_words.sort_values(
        ["covariate", "fdr_q_value", "p_value"],
        ascending=[True, True, True],
    )

    lines: list[str] = []
    lines.append("Part C Result Snapshot Run 3 (Experiment 2: Spearman + Covariates)")
    lines.append("")
    lines.append("Data prep summary")
    lines.append(
        f"- states in merged analysis: {n_states} | topics: {n_topics} | "
        f"total Spearman tests: {n_tests}"
    )
    lines.append(
        f"- significant after FDR: income={sig_income}, education={sig_education}, "
        f"total={len(sig)}"
    )
    lines.append(
        f"- income-education Spearman across included states: "
        f"rho={income_edu_rho:.3f}, p={income_edu_p:.4g}"
    )
    lines.append("")
    lines.append("Top |rho| topics for income")
    lines.append(
        top_income[
            ["topic", "top_words", "spearman_rho", "p_value", "fdr_q_value", "significant_fdr"]
        ].to_string(index=False)
    )
    lines.append("")
    lines.append("Top |rho| topics for education")
    lines.append(
        top_education[
            ["topic", "top_words", "spearman_rho", "p_value", "fdr_q_value", "significant_fdr"]
        ].to_string(index=False)
    )
    lines.append("")
    lines.append("FDR-significant topic-covariate links")
    lines.append(
        sig_with_words[
            ["covariate", "topic", "top_words", "spearman_rho", "p_value", "fdr_q_value"]
        ].to_string(index=False)
    )

    text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Spearman covariate analysis for Experiment 2."
    )
    parser.add_argument(
        "--state-topic-with-covariates",
        type=Path,
        default=Path("results/statistics/state_topic_with_covariates.csv"),
    )
    parser.add_argument(
        "--spearman",
        type=Path,
        default=Path("results/statistics/spearman_topic_covariates.csv"),
    )
    parser.add_argument(
        "--spearman-significant",
        type=Path,
        default=Path("results/statistics/spearman_topic_covariates_significant.csv"),
    )
    parser.add_argument(
        "--representative-prompts",
        type=Path,
        default=Path("src/topic_modeling/results/representative_prompts.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/statistics/part_c_results/part_c_result_run3.txt"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_experiment2_snapshot(
        state_topic_with_covariates_path=args.state_topic_with_covariates,
        spearman_path=args.spearman,
        spearman_sig_path=args.spearman_significant,
        representative_prompts_path=args.representative_prompts,
        output_path=args.output,
    )
    print(f"Saved run3 summary to {args.output}")
