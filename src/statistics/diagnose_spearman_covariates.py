from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        rho, _ = spearmanr(x[idx], y[idx])
        samples[i] = rho
    low = float(np.quantile(samples, alpha / 2))
    high = float(np.quantile(samples, 1 - alpha / 2))
    return low, high


def run_deep_analysis(
    merged_path: Path,
    spearman_path: Path,
    sig_path: Path,
    rep_path: Path,
    output_txt: Path,
    output_csv: Path,
    n_boot: int = 4000,
    seed: int = 42,
) -> None:
    merged = pd.read_csv(merged_path)
    sp = pd.read_csv(spearman_path)
    sig = pd.read_csv(sig_path)
    rep = pd.read_csv(rep_path)[["topic_id", "top_words"]].drop_duplicates("topic_id")
    rep["topic"] = rep["topic_id"].apply(lambda x: f"topic_{int(x)}")
    topic_words = dict(zip(rep["topic"], rep["top_words"]))

    income = sp[sp["covariate"] == "income"].set_index("topic")["spearman_rho"]
    education = sp[sp["covariate"] == "education"].set_index("topic")["spearman_rho"]
    common = income.index.intersection(education.index)

    effect_corr = float(np.corrcoef(income.loc[common], education.loc[common])[0, 1])
    sign_consistency = float((np.sign(income.loc[common]) == np.sign(education.loc[common])).mean())

    overlap = {}
    for k in [5, 10, 20]:
        ti = set(
            sp[sp["covariate"] == "income"]
            .assign(abs_rho=lambda d: d["spearman_rho"].abs())
            .sort_values("abs_rho", ascending=False)
            .head(k)["topic"]
        )
        te = set(
            sp[sp["covariate"] == "education"]
            .assign(abs_rho=lambda d: d["spearman_rho"].abs())
            .sort_values("abs_rho", ascending=False)
            .head(k)["topic"]
        )
        inter = len(ti & te)
        union = len(ti | te)
        overlap[k] = (inter, union, inter / union if union else np.nan)

    gap = pd.DataFrame(
        {
            "topic": common,
            "rho_income": income.loc[common].values,
            "rho_education": education.loc[common].values,
        }
    )
    gap["abs_gap"] = (gap["rho_income"] - gap["rho_education"]).abs()
    gap["top_words"] = gap["topic"].map(topic_words).fillna("")
    gap = gap.sort_values("abs_gap", ascending=False).head(10)

    sig_out = sig.copy()
    sig_out["top_words"] = sig_out["topic"].map(topic_words).fillna("")
    sig_out["nonzero_states"] = sig_out["topic"].apply(lambda t: int((merged[t] > 0).sum()))
    sig_out["max_topic_prop"] = sig_out["topic"].apply(lambda t: float(merged[t].max()))
    ci_low = []
    ci_high = []
    for _, row in sig_out.iterrows():
        cov = row["covariate"]
        topic = row["topic"]
        x = merged[cov].to_numpy()
        y = merged[topic].to_numpy()
        low, high = _bootstrap_spearman_ci(x, y, n_boot=n_boot, seed=seed)
        ci_low.append(low)
        ci_high.append(high)
    sig_out["rho_ci95_low"] = ci_low
    sig_out["rho_ci95_high"] = ci_high

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    sig_out.to_csv(output_csv, index=False)

    lines = []
    lines.append("Part C Result Snapshot Run 4 (Experiment 2 Deep-Dive)")
    lines.append("")
    lines.append("Effect-structure diagnostics")
    lines.append(f"- correlation between income-rho and education-rho vectors: {effect_corr:.3f}")
    lines.append(f"- sign consistency across topics: {sign_consistency:.3f}")
    for k in [5, 10, 20]:
        inter, union, jac = overlap[k]
        lines.append(f"- top-{k} overlap (income vs education): {inter}/{union} (Jaccard={jac:.3f})")
    lines.append("")
    lines.append("Top 10 topics with largest rho gap between income and education")
    lines.append(
        gap[["topic", "top_words", "rho_income", "rho_education", "abs_gap"]].to_string(index=False)
    )
    lines.append("")
    lines.append("Significant links with nonzero-state support + bootstrap CI")
    lines.append(
        sig_out[
            [
                "covariate",
                "topic",
                "top_words",
                "spearman_rho",
                "fdr_q_value",
                "nonzero_states",
                "max_topic_prop",
                "rho_ci95_low",
                "rho_ci95_high",
            ]
        ].to_string(index=False)
    )
    output_txt.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-dive diagnostics for Spearman Experiment 2.")
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
        "--output-txt",
        type=Path,
        default=Path("results/statistics/part_c_results/part_c_result_run4.txt"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/statistics/part_c_results/spearman_significant_deep_summary.csv"),
    )
    parser.add_argument("--n-boot", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_deep_analysis(
        merged_path=args.state_topic_with_covariates,
        spearman_path=args.spearman,
        sig_path=args.spearman_significant,
        rep_path=args.representative_prompts,
        output_txt=args.output_txt,
        output_csv=args.output_csv,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(f"Saved deep-dive results to {args.output_txt} and {args.output_csv}")
