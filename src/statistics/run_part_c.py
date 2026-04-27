from __future__ import annotations

import argparse
from pathlib import Path

from state_topic_correlations import run_state_topic_analysis
from topic_group_significance import run_significance_tests
from weighted_log_odds import run_log_odds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all Part C analyses using state information only."
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
        help="Path with state-labeled prompts for per-state document counts.",
    )
    parser.add_argument(
        "--log-odds-input",
        type=Path,
        default=Path("data/wildchat_log_odds.parquet"),
        help="Path to cleaned text parquet for log-odds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/statistics"),
        help="Directory to store all outputs.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold.")
    parser.add_argument("--n-perm", type=int, default=5000, help="Permutation samples per topic.")
    parser.add_argument("--n-boot", type=int, default=5000, help="Bootstrap samples per topic.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--min-total-count",
        type=int,
        default=20,
        help="Minimum token count for weighted log-odds vocabulary.",
    )
    parser.add_argument(
        "--min-docs-per-state",
        type=int,
        default=100,
        help="Minimum prompts required to include a state in log-odds.",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=3,
        help="Minimum token length for log-odds tokenization.",
    )
    parser.add_argument(
        "--log-odds-state-whitelist",
        type=Path,
        default=Path("data/state_topic_proportions.parquet"),
        help=(
            "Parquet with a 'state' column used to restrict log-odds to the same "
            "state set as topic-level analyses."
        ),
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top/bottom words to export.")
    parser.add_argument(
        "--top-k-per-topic",
        type=int,
        default=5,
        help="Top/bottom states per topic to export.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_state_topic_analysis(
        topic_proportions_path=args.topic_proportions,
        output_dir=args.output_dir,
        top_k_per_topic=args.top_k_per_topic,
    )
    run_significance_tests(
        topic_proportions_path=args.topic_proportions,
        state_docs_input_path=args.state_docs_input,
        output_dir=args.output_dir,
        n_perm=args.n_perm,
        seed=args.seed,
        alpha=args.alpha,
    )
    run_log_odds(
        log_odds_input_path=args.log_odds_input,
        output_dir=args.output_dir,
        min_total_count=args.min_total_count,
        top_k=args.top_k,
        min_docs_per_state=args.min_docs_per_state,
        min_token_length=args.min_token_length,
        state_whitelist_path=args.log_odds_state_whitelist,
    )
    print(f"Completed Part C analysis. Outputs are in: {args.output_dir}")
