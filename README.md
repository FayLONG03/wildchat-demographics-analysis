# NLP for CSS — Final Course Project

This is the final course project for NLP for CSS. This project contains four sections.

---

## Section 1: Data Preprocessing

### Preprocessing steps applied to both topic modeling and word statistics:

- **Pre-filtering:** Prefiltered for rows that are English, US-based, and with a non-empty `state` field.
- **Extracting the first user turn:** Only the first block of content with `role = "user"` is kept. Many relevant papers use the first user turn rather than the full conversations for analysis, since follow-up turns are largely reactive, containing clarifications, corrections, or continuations that are responses to the AI rather than independent expressions of purpose. For our analysis of why people use AI, the first user turn is the cleanest signal.
- **Deduplication:** Upon manual investigation of the dataset, it was discovered that many conversations originated from queries with identical or highly similar prompts. For example, a user might send slightly modified prompts 10 times asking the AI to write a school essay, but this behavior shouldn't contribute to the "school essay" topic 10 times. To address this, deduplication using MinHash was applied, removing near-identical user inputs from the same `hashed_ip` (used as a unique user identifier).

### Preprocessing step applied to word statistics only:

- **Lowercasing and punctuation removal:** Common practice for word statistics preprocessing.

### Preprocessing steps applied to topic modeling only:

- **Removing states with small sample sizes:** Removed states with fewer than 100 conversations and fewer than 40 unique users.
- **Truncating user texts with tokens > 512:** Only the first 250 tokens and last 250 tokens are kept for user prompts longer than 512 tokens. This decision was made because the goal of this project is to analyze users' purposes for using AI, and people usually state their purposes at the beginning and end of their prompts.

### Limitations

> The `language` column of the original dataset is not completely correct. A small proportion of texts in other languages or in mixed languages are incorrectly marked as English. With that being said, this minor issue is unlikely to impact the analysis results.

---

## Section 2: Topic Modeling

### Model: BERTopic

We apply BERTopic (Grootendorst, 2022) to the preprocessed user turns. 
Documents are encoded using `all-MiniLM-L6-v2`, reduced with UMAP, 
clustered with HDBSCAN, and keywords extracted via class-based TF-IDF.

### Parameter sweep

We swept over 5 configurations varying `min_topic_size` (50, 100, 200) 
and UMAP `n_neighbors` (10, 15). Each run was evaluated on topic coherence 
(c_v), topic diversity, topic count, and outlier rate.

### Best configuration

| Parameter | Value |
|---|---|
| min_topic_size | 100 |
| n_neighbors | 15 |
| min_cluster_size | 100 |
| Topics found | 76 |
| Coherence (c_v) | 0.5806 |
| Diversity | 0.8408 |
| Outlier rate | 44.9% |

### Outputs
- `results/bertopic_experiment_log.csv` — scores for all 5 runs
- `results/representative_prompts.csv` — top prompts per topic for labeling
- `results/state_topic_proportions.parquet` — state-level topic proportions for correlation analysis

---

## Section 3: Word Statistics

Section 3 now includes a reproducible Part C pipeline for:

1. state-topic concentration and ranking summaries,
2. weighted log-odds comparisons for each state vs. the rest of states,
3. permutation-based significance tests of topic heterogeneity across states.

### Required input data

- `data/state_topic_proportions.parquet`
- `data/wildchat_bertopic.parquet`
- `data/wildchat_log_odds.parquet`

### Run all Part C analyses

```bash
python src/statistics/run_part_c.py \
  --topic-proportions data/state_topic_proportions.parquet \
  --state-docs-input data/wildchat_bertopic.parquet \
  --log-odds-input data/wildchat_log_odds.parquet \
  --output-dir results/statistics
```

### Output files

- `results/statistics/state_topic_long.csv`
- `results/statistics/topic_variation_by_state.csv`
- `results/statistics/state_topic_top_5_states.csv`
- `results/statistics/state_topic_bottom_5_states.csv`
- `results/statistics/topic_state_heterogeneity_significance.csv`
- `results/statistics/topic_state_heterogeneity_significant.csv`
- `results/statistics/log_odds_state_vs_rest_top_50.csv`
- `results/statistics/log_odds_state_vs_rest_bottom_50.csv`
- `results/statistics/log_odds_<STATE>_vs_rest.csv` for each included state

---

## Section 4: Qualitative Analysis
