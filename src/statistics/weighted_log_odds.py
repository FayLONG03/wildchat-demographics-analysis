from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd

from utils import ensure_output_dir

TOKEN_PATTERN = re.compile(r"[a-z]{3,}")
STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "your", "with", "that", "this", "was",
    "have", "has", "had", "from", "they", "their", "them", "his", "her", "she", "him", "its",
    "our", "out", "all", "any", "can", "may", "might", "will", "would", "should", "could",
    "into", "onto", "over", "under", "than", "then", "there", "here", "where", "when", "what",
    "who", "whom", "why", "how", "about", "after", "before", "between", "because", "while",
    "also", "just", "very", "more", "most", "some", "such", "only", "much", "many", "each",
    "other", "another", "these", "those", "been", "being", "were", "is", "am", "to", "of", "in",
    "on", "at", "by", "as", "it", "or", "an", "a", "if", "we", "us", "do", "did", "does",
    "done", "i", "me", "my", "mine", "he", "hers", "theirs", "ours", "yours"
}


def iter_tokens(text_series: pd.Series, stopwords: set[str], min_token_length: int):
    for text in text_series.dropna():
        for token in TOKEN_PATTERN.findall(str(text).lower()):
            if len(token) < min_token_length:
                continue
            if token in stopwords:
                continue
            if token.isnumeric():
                continue
            if token:
                yield token


def count_tokens(text_series: pd.Series, stopwords: set[str], min_token_length: int) -> Counter:
    counter = Counter()
    for token in iter_tokens(text_series, stopwords=stopwords, min_token_length=min_token_length):
        counter[token] += 1
    return counter


def weighted_log_odds(
    group_a_counts: Counter,
    group_b_counts: Counter,
    prior_counts: Counter,
    min_total_count: int = 20,
) -> pd.DataFrame:
    vocab = set(prior_counts) | set(group_a_counts) | set(group_b_counts)
    n_a = sum(group_a_counts.values())
    n_b = sum(group_b_counts.values())
    alpha_0 = sum(prior_counts.values())

    rows = []
    for token in vocab:
        y_a = group_a_counts.get(token, 0)
        y_b = group_b_counts.get(token, 0)
        alpha_w = prior_counts.get(token, 0)
        total = y_a + y_b
        if total < min_total_count:
            continue
        if alpha_w <= 0:
            continue

        # Monroe et al. (2008) weighted log-odds with an informative Dirichlet prior.
        delta = np.log((y_a + alpha_w) / (n_a + alpha_0 - y_a - alpha_w)) - np.log(
            (y_b + alpha_w) / (n_b + alpha_0 - y_b - alpha_w)
        )
        variance = 1.0 / (y_a + alpha_w) + 1.0 / (y_b + alpha_w)
        z_score = delta / np.sqrt(variance)

        rows.append(
            {
                "token": token,
                "count_state": y_a,
                "count_rest": y_b,
                "count_total": total,
                "log_odds_delta": delta,
                "z_score": z_score,
            }
        )

    result = pd.DataFrame(rows).sort_values("z_score", ascending=False)
    return result


def run_log_odds(
    log_odds_input_path: Path,
    output_dir: Path,
    min_total_count: int = 20,
    top_k: int = 50,
    min_docs_per_state: int = 100,
    min_token_length: int = 3,
    state_whitelist_path: Path | None = None,
) -> None:
    ensure_output_dir(output_dir)

    text_df = pd.read_parquet(log_odds_input_path).copy()
    text_df["state"] = text_df["state"].astype(str).str.strip()
    text_df = text_df[text_df["state"].str.len() > 0].copy()

    state_doc_counts = text_df["state"].value_counts()
    valid_states = sorted(state_doc_counts[state_doc_counts >= min_docs_per_state].index.tolist())
    if not valid_states:
        raise ValueError(
            f"No states have at least {min_docs_per_state} documents in {log_odds_input_path}."
        )

    if state_whitelist_path is not None:
        whitelist_df = pd.read_parquet(state_whitelist_path)
        if "state" not in whitelist_df.columns:
            raise ValueError(f"'state' column not found in {state_whitelist_path}")
        whitelist_states = set(whitelist_df["state"].astype(str).str.strip())
        valid_states = sorted(set(valid_states).intersection(whitelist_states))
        if not valid_states:
            raise ValueError(
                "No valid states remain after applying state whitelist. "
                "Please check the whitelist file and state names."
            )

    filtered = text_df[text_df["state"].isin(valid_states)].copy()
    prior_counts = count_tokens(filtered["user_text"], stopwords=STOPWORDS, min_token_length=min_token_length)

    # Remove stale per-state files from previous runs (e.g., when state set changes).
    valid_slug_set = {state.replace(" ", "_") for state in valid_states}
    for existing_file in output_dir.glob("log_odds_*_vs_rest.csv"):
        name = existing_file.name
        if name in {"log_odds_state_vs_rest_top_50.csv", "log_odds_state_vs_rest_bottom_50.csv"}:
            continue
        state_slug = name[len("log_odds_") : -len("_vs_rest.csv")]
        if state_slug not in valid_slug_set:
            existing_file.unlink(missing_ok=True)

    top_rows = []
    bottom_rows = []
    for state in valid_states:
        state_df = filtered[filtered["state"] == state]
        rest_df = filtered[filtered["state"] != state]

        state_counts = count_tokens(
            state_df["user_text"],
            stopwords=STOPWORDS,
            min_token_length=min_token_length,
        )
        rest_counts = count_tokens(
            rest_df["user_text"],
            stopwords=STOPWORDS,
            min_token_length=min_token_length,
        )
        result = weighted_log_odds(
            group_a_counts=state_counts,
            group_b_counts=rest_counts,
            prior_counts=prior_counts,
            min_total_count=min_total_count,
        )
        result["state"] = state
        result["comparison"] = f"{state}_vs_rest"
        result["state_docs_n"] = len(state_df)
        result["rest_docs_n"] = len(rest_df)

        state_slug = state.replace(" ", "_")
        result.to_csv(output_dir / f"log_odds_{state_slug}_vs_rest.csv", index=False)
        top_rows.append(result.head(top_k))
        bottom_rows.append(result.tail(top_k).sort_values("z_score"))

    pd.concat(top_rows, ignore_index=True).to_csv(
        output_dir / f"log_odds_state_vs_rest_top_{top_k}.csv",
        index=False,
    )
    pd.concat(bottom_rows, ignore_index=True).to_csv(
        output_dir / f"log_odds_state_vs_rest_bottom_{top_k}.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run weighted log-odds on user text for each state vs all other states."
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
        help="Directory to store log-odds outputs.",
    )
    parser.add_argument(
        "--min-total-count",
        type=int,
        default=20,
        help="Minimum combined token count across groups to include token.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="How many top/bottom words to save per state.",
    )
    parser.add_argument(
        "--min-docs-per-state",
        type=int,
        default=100,
        help="Minimum number of prompts required for a state to be included.",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=3,
        help="Minimum token length after regex tokenization.",
    )
    parser.add_argument(
        "--state-whitelist",
        type=Path,
        default=None,
        help=(
            "Optional parquet file with a 'state' column used to restrict log-odds "
            "to a specific state set."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_log_odds(
        log_odds_input_path=args.log_odds_input,
        output_dir=args.output_dir,
        min_total_count=args.min_total_count,
        top_k=args.top_k,
        min_docs_per_state=args.min_docs_per_state,
        min_token_length=args.min_token_length,
        state_whitelist_path=args.state_whitelist,
    )
    print(f"Saved weighted log-odds results to {args.output_dir}")
