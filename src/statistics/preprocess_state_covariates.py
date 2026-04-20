from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _to_numeric(value: object) -> float | None:
    if pd.isna(value):
        return None
    cleaned = str(value).replace(",", "").replace("%", "").strip()
    if cleaned in {"", "-", "(X)", "N", "**", "***"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_income(inc_df: pd.DataFrame) -> pd.DataFrame:
    label_col = inc_df.columns[0]
    labels = inc_df[label_col].astype(str).str.strip()
    target_idx = labels[labels == "Median income (dollars)"].index
    if len(target_idx) == 0:
        raise ValueError("Could not find 'Median income (dollars)' row in income table.")
    row = inc_df.loc[target_idx[0]]

    rows: list[dict[str, object]] = []
    for col in inc_df.columns[1:]:
        parts = str(col).split("!!")
        if len(parts) < 3:
            continue
        state, group, measure = parts[0], parts[1], parts[2]
        if state == "Puerto Rico":
            continue
        if group != "Households" or measure != "Estimate":
            continue
        value = _to_numeric(row[col])
        if value is None:
            continue
        rows.append({"state": state, "income": value})
    return pd.DataFrame(rows)


def _extract_education(edu_df: pd.DataFrame) -> pd.DataFrame:
    label_col = edu_df.columns[0]
    labels = edu_df[label_col].astype(str).str.strip()

    start_idx_series = labels[labels == "Population 25 years and over"].index
    if len(start_idx_series) == 0:
        raise ValueError("Could not find 'Population 25 years and over' block in education table.")
    start_idx = int(start_idx_series[0])

    next_population_idxs = labels[(labels.index > start_idx) & labels.str.startswith("Population ")].index
    end_idx = int(next_population_idxs[0]) if len(next_population_idxs) > 0 else len(labels)

    block = labels.iloc[start_idx + 1 : end_idx]
    target_in_block = block[block == "Bachelor's degree or higher"].index
    if len(target_in_block) == 0:
        raise ValueError(
            "Could not find 'Bachelor's degree or higher' row inside "
            "'Population 25 years and over' block."
        )
    row = edu_df.loc[int(target_in_block[0])]

    rows: list[dict[str, object]] = []
    for col in edu_df.columns[1:]:
        parts = str(col).split("!!")
        if len(parts) < 3:
            continue
        state, group, measure = parts[0], parts[1], parts[2]
        if state == "Puerto Rico":
            continue
        if group != "Percent" or measure != "Estimate":
            continue
        value = _to_numeric(row[col])
        if value is None:
            continue
        rows.append({"state": state, "education": value})
    return pd.DataFrame(rows)


def build_covariates(income_path: Path, education_path: Path) -> pd.DataFrame:
    income_df = pd.read_csv(income_path)
    education_df = pd.read_csv(education_path)

    income = _extract_income(income_df)
    education = _extract_education(education_df)

    covariates = income.merge(education, on="state", how="inner")
    covariates = covariates.drop_duplicates(subset=["state"]).sort_values("state")
    return covariates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess ACS state-level income/education tables into a clean covariate CSV."
    )
    parser.add_argument(
        "--income",
        type=Path,
        default=Path("data/Income/ACSST1Y2024.S1901-2026-04-20T174538.csv"),
        help="Path to ACS S1901 wide-format CSV.",
    )
    parser.add_argument(
        "--education",
        type=Path,
        default=Path("data/Education/ACSST1Y2024.S1501-2026-04-20T174348.csv"),
        help="Path to ACS S1501 wide-format CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/state_covariates.csv"),
        help="Output CSV path for cleaned state covariates.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    covariates = build_covariates(args.income, args.education)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    covariates.to_csv(args.output, index=False)
    print(f"Saved {len(covariates)} states to {args.output}")
