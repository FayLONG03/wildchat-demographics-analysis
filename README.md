# NLP for CSS — Final Course Project

This is the final course project for NLP for CSS. This project contains four sections.

---

## Section 1: Data Preprocessing

### Preprocessing steps applied to both topic modeling and word statistics:

- **Pre-filtering:** Prefiltered for rows that are English, US-based, and with a non-empty `state` field.
- **Extracting the first user turn:** Only the first block of content with `role = "user"` is kept. Many relevant papers use the first user turn rather than the full conversations for analysis, since follow-up turns are largely reactive, containing clarifications, corrections, or continuations that are responses to the AI rather than independent expressions of purpose. For our analysis of why people use AI, the first user turn is the cleanest signal.
- **Deduplication:** Upon manual investigation of the dataset, it was discovered that many conversations originated from queries with identical or highly similar prompts. For example, a user might send slightly modified prompts 10 times asking the AI to write a school essay, but this behavior shouldn't contribute to the "school essay" topic 10 times. To address this, deduplication using MinHash was applied, removing near-identical user inputs from the same `hashed_ip` (used as a unique user identifier).
- **Removing states with small sample sizes:** Removed states with fewer than 100 conversations and fewer than 40 unique users.

### Preprocessing step applied to word statistics only:

- **Lowercasing and punctuation removal:** Common practice for word statistics preprocessing.

### Preprocessing steps applied to topic modeling only:

- **Truncating user texts with tokens > 512:** Only the first 250 tokens and last 250 tokens are kept for user prompts longer than 512 tokens. This decision was made because the goal of this project is to analyze users' purposes for using AI, and people usually state their purposes at the beginning and end of their prompts.

### Limitations

> The `language` column of the original dataset is not completely correct. A small proportion of texts in other languages or in mixed languages are incorrectly marked as English. With that being said, this minor issue is unlikely to impact the analysis results.

---

## Section 2: Topic Modeling

### 2.1 BERTopic (`src/topic_modeling/bertopic_wildchat.ipynb`)

We apply BERTopic (Grootendorst, 2022) to the preprocessed user turns.
Documents are encoded using `all-MiniLM-L6-v2`, reduced with UMAP,
clustered with HDBSCAN, and keywords extracted via class-based TF-IDF.

**Parameter sweep:** We swept over 5 configurations varying `min_topic_size` (50, 100, 200)
and UMAP `n_neighbors` (10, 15). Each run was evaluated on topic coherence (c_v), topic
diversity, topic count, and outlier rate.

**Best configuration (run_04):**

| Parameter | Value |
|---|---|
| min_topic_size | 100 |
| n_neighbors | 15 |
| min_cluster_size | 100 |
| Topics found | 76 |
| Coherence (c_v) | 0.5806 |
| Diversity | 0.8408 |
| Outlier rate | 44.9% |

Documents not assigned to any topic (44.9%) are excluded from downstream analyses.

**Outputs:**
- `src/topic_modeling/results/bertopic_experiment_log.csv` — scores for all 5 runs
- `src/topic_modeling/results/representative_prompts.csv` — top prompts per topic for LLM-assisted labeling
- `src/topic_modeling/results/representative_prompts_labeled.csv` — labeled prompts with topic labels and super-topic assignments
- `src/topic_modeling/results/representative_prompts_labeled_with_super_topics.csv` — final labeled prompts with super-topic taxonomy
- `src/topic_modeling/results/topic_level_super_topics.csv` — one row per topic with its label and super-topic category
- `data/state_topic_proportions.parquet` — state × topic proportion matrix (76 raw topics; `*.parquet` is gitignored)

### 2.2 Exploratory Log-Odds Analysis (`src/topic_modeling/log_odds_analysis.ipynb`)

As an exploratory step, we compute weighted log-odds ratios (Monroe et al., 2008) to compare
topic usage between high and low demographic groups. States are split into top and bottom
quartiles by education (% bachelor's degree+) and income (median household income); the
middle two quartiles are excluded to sharpen group contrasts.

The weighted log-odds formula uses a Dirichlet prior (α = 0.01) to account for variance in
low-frequency topics. A permutation test (1,000 iterations) was also run on the top education
topics as an additional robustness check.

**Inputs:**
- `data/state_topic_proportions.parquet`
- `data/state_covariates.csv`

**Outputs:**
- `src/topic_modeling/results/log_odds_education.csv` — log-odds z-scores for all 76 topics (education)
- `src/topic_modeling/results/log_odds_income.csv` — log-odds z-scores for all 76 topics (income)
- `src/topic_modeling/results/log_odds_figures/log_odds_education.png` — bar chart visualization
- `src/topic_modeling/results/log_odds_figures/log_odds_income.png` — bar chart visualization
- `src/topic_modeling/results/permutation_test_education.csv` — permutation test results for top education topics

> Note: The final log-odds figures used in the paper (Figure 7) are produced in `src/analyisis/topic_demographic_relationship.ipynb` using the merged 71-topic set, not this notebook.

### 2.3 CTM Robustness Check (`src/topic_modeling/ctm_robustness.ipynb`)

We fit CombinedTM (Bianchi et al., 2021) on the same 68,668 preprocessed documents
to evaluate whether BERTopic results are robust to the choice of topic model. CTM uses
the same `all-MiniLM-L6-v2` embeddings for a fair comparison, with 76 components and 100
training epochs. Training takes approximately 11 hours on CPU.

**Inputs:**
- `data/wildchat_bertopic.parquet`

**Results:**

| Model | Topics | Coherence (c_v) | Diversity |
|---|---|---|---|
| BERTopic (run_04) | 76 | 0.5806 | 0.8408 |
| CombinedTM | 76 | 0.4821 | 0.6132 |

BERTopic outperforms CombinedTM on both metrics, likely because CTM's bag-of-words
component is less well-suited to short, heterogeneous conversational text.

**Outputs:**
- `src/topic_modeling/results/ctm_topic_words.csv` — top 10 words per CTM topic
- `src/topic_modeling/results/ctm_topic_words_labeled.csv` — CTM topics with labels
- `src/topic_modeling/results/ctm_vs_bertopic_comparison.csv` — side-by-side coherence/diversity comparison
- `src/topic_modeling/results/ctm_state_topic_proportions.parquet` — state-level CTM topic proportions (`*.parquet` is gitignored)

> The Jaccard lexical alignment analysis and demographic effect size comparison used in Appendix D of the paper are computed in `src/analyisis/CTM_robustness.ipynb` using the pre-computed outputs above.

---

## Section 3: Word Statistics

Section 3 includes a reproducible Part C word statistics pipeline for:

1. state-topic concentration and ranking summaries,
2. weighted log-odds comparisons for each state vs. the rest of states,
3. simulation-based significance tests of topic heterogeneity across states (binomial null model weighted by per-state document counts; see `src/statistics/topic_group_significance.py`),
4. (extension) Spearman analyses with external ACS covariates (income + education), plus summary/diagnostic snapshots.

### Core Part C input data

- `data/state_topic_proportions.parquet`
- `data/wildchat_bertopic.parquet`
- `data/wildchat_log_odds.parquet`

### Run all Part C analyses

```bash
python src/statistics/run_part_c.py \
  --topic-proportions data/state_topic_proportions.parquet \
  --state-docs-input data/wildchat_bertopic.parquet \
  --log-odds-input data/wildchat_log_odds.parquet \
  --log-odds-state-whitelist data/state_topic_proportions.parquet \
  --min-token-length 3 \
  --output-dir results/statistics
```

### Core output files (`results/statistics`)

- `results/statistics/state_topic_long.csv`
- `results/statistics/topic_variation_by_state.csv`
- `results/statistics/state_topic_top_5_states.csv`
- `results/statistics/state_topic_bottom_5_states.csv`
- `results/statistics/topic_state_heterogeneity_significance.csv`
- `results/statistics/topic_state_heterogeneity_significant.csv`
- `results/statistics/log_odds_state_vs_rest_top_50.csv`
- `results/statistics/log_odds_state_vs_rest_bottom_50.csv`
- `results/statistics/log_odds_<STATE>_vs_rest.csv` for each included state

### Covariate extension (income + education)

These scripts are run separately from `run_part_c.py`:

1) Build covariates from ACS tables

```bash
python src/statistics/preprocess_state_covariates.py \
  --income data/Income/ACSST1Y2024.S1901-2026-04-20T174538.csv \
  --education data/Education/ACSST1Y2024.S1501-2026-04-20T174348.csv \
  --output data/state_covariates.csv
```

2) Run Spearman correlations with FDR correction

```bash
python src/statistics/spearman_with_covariates.py \
  --topic-proportions data/state_topic_proportions.parquet \
  --covariates data/state_covariates.csv \
  --output-dir results/statistics
```

3) Generate report-ready snapshots

```bash
python src/statistics/summarize_spearman_covariates.py
python src/statistics/diagnose_spearman_covariates.py
```

Additional outputs from this extension:

- `data/state_covariates.csv`
- `results/statistics/state_topic_with_covariates.csv`
- `results/statistics/spearman_topic_covariates.csv`
- `results/statistics/spearman_topic_covariates_significant.csv`
- `results/statistics/part_c_results/part_c_result_run3.txt`
- `results/statistics/part_c_results/part_c_result_run4.txt`
- `results/statistics/part_c_results/spearman_significant_deep_summary.csv`

---

## Section 4: Analysis

- **`src/analyisis/labeled_topics.ipynb`** — Summarizes BERTopic output: topic labeling, cross-state heterogeneity test (binomial null model), label merging, and weighted prevalence at the topic and super-topic level.
- **`src/analyisis/topic_demographic_relationship.ipynb`** — Computes Spearman's ρ between merged topic proportions and state-level income / education, with bootstrap confidence intervals, heatmaps, scatter plots, and log-odds cross-validation.
- **`src/analyisis/CTM_robustness.ipynb`** — Validates BERTopic findings by comparing merged topic word sets against a CombinedTM (CTM) model using Jaccard similarity, and contrasts model-level coherence and diversity scores (scores pre-computed in `src/topic_modeling/ctm_robustness.ipynb`).
- **`src/summary.ipynb`** — Executive summary displaying key figures and conclusions from the three analysis notebooks above.

---

## Project Structure

```
.
├── data/                                   # Input data (large *.parquet files are gitignored)
│   ├── Education/                          # ACS education data by state
│   ├── Income/                             # ACS income data by state
│   ├── state_covariates.csv                # Merged state-level demographics
│   ├── state_topic_proportions.parquet     # State × topic proportion matrix
│   ├── wildchat_bertopic.parquet           # Preprocessed corpus with BERTopic labels
│   └── wildchat_log_odds.parquet           # Preprocessed corpus for log-odds
│
├── src/
│   ├── preprocess/                         # Preprocessing notebooks
│   │   ├── preprocessing_for_bertopic.ipynb
│   │   └── proprocessing_for_log_odds.ipynb
│   ├── topic_modeling/                     # Topic model training and evaluation
│   │   ├── bertopic_wildchat.ipynb         # BERTopic parameter sweep and best-run selection
│   │   ├── ctm_robustness.ipynb            # CTM training and metric computation
│   │   ├── log_odds_analysis.ipynb         # Log-odds computation per state
│   │   └── results/                        # Intermediate model outputs (labels, word lists)
│   ├── statistics/                         # Part C scripts (core pipeline + ACS/Spearman extensions)
│   ├── analyisis/                          # Analysis notebooks (Section 4)
│   │   ├── labeled_topics.ipynb
│   │   ├── topic_demographic_relationship.ipynb
│   │   └── CTM_robustness.ipynb
│   └── summary.ipynb                       # Executive summary (Section 4)
│
├── results/
│   ├── statistics/                         # Part C outputs (core + covariate extension)
│   ├── statistics_merged71/                # Optional merged-topic statistics outputs
│   ├── qualitative/                        # Intermediate tables and figures from earlier runs
│   └── summary/                            # Figures and tables written by Section 4 notebooks
│       ├── labeled_topics/
│       ├── topic_demographic_relationship/
│       └── CTM_robustness/
│
├── requirements.txt
└── README.md
```

---


