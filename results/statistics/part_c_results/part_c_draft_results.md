# Part C Draft: Quantitative Results (State-Only)

## 1) State-level heterogeneity in topic usage

Using `state_topic_proportions.parquet` (36 states, 76 topics), we tested whether topic prevalence varies across states more than expected under a null where each state draws prompts from the same global topic rate (while preserving state document counts). We ran permutation tests topic-by-topic and controlled for multiple testing using Benjamini-Hochberg FDR.

All 76 topics were significant after FDR correction (all `q ≈ 0.0002` with 5000 permutations). This indicates broad geographic heterogeneity in topic prevalence.

Top high-variation topics by state-level dispersion included:

- `topic_7` (`fish, pond, fart, story, comedic, vividly, farts, water`) — highest in **Massachusetts**.
- `topic_0` (`script, write script, vs, state, round, conference, write, pm`) — highest in **Pennsylvania**.
- `topic_9` (`email, bank, security, funds, presidio, anonymized, payment`) — highest in **Florida**.
- `topic_1` (`natsuki, monika, sayori, yuri, mc, just, club, literature`) — highest in **Michigan**.
- `topic_2` (`translate, english, chinese, japanese, mean, spanish`) — highest in **Connecticut**.

These results support the claim that usage intents (as represented by BERTopic clusters) are not evenly distributed across states.

## 2) State-overindexing patterns

To make geographic differences interpretable, we ranked the top 5 and bottom 5 states for each topic (`state_topic_top_5_states.csv`, `state_topic_bottom_5_states.csv`) and computed z-scores within topic (`state_topic_long.csv`).

Examples:

- `topic_0` is strongly concentrated in **Pennsylvania** (`topic_proportion = 0.705`, z-score `= 5.90`).
- `topic_7` is strongly concentrated in **Massachusetts** (`topic_proportion = 0.739`, z-score `= 5.91`).
- `topic_1` is strongly concentrated in **Michigan** (`topic_proportion = 0.501`, z-score `= 5.91`).
- `topic_9` is strongly concentrated in **Florida** (`topic_proportion = 0.510`).

This ranking output gives direct evidence for “where” particular usage patterns are over-represented.

## 3) Weighted log-odds (state vs. rest)

We ran weighted log-odds on cleaned user text (`wildchat_log_odds.parquet`) for each state against all remaining states, using an informative prior and minimum token-frequency threshold. Tokenization used lowercase regex filtering with stopword removal and minimum token length constraints to improve interpretability. To keep analyses consistent across Part C, we restricted log-odds to the same state set used by topic-level analysis (`state_topic_proportions.parquet`), resulting in 36 states.

Illustrative lexical over-indexing:

- **Arizona**: `hair`, `hairpin`, `clip`, `comb`, `piercing`, `headband`, `makeup`
- **Pennsylvania**: `script`, `stadium`, `conference`, `state`, `round`, `bowl`
- **Massachusetts**: `fish`, `pond`, `fart`, `comedic`, `vividly`
- **Florida**: `email`, `bank`, `cameron`, `funds`, `payment`
- **Michigan**: `natsuki`, `monika`, `sayori`, `yuri`, `clubroom`

These lexical signatures align with the topic-level state concentration findings and provide interpretable language-level evidence for regional differentiation.

## 4) Interpretation framing for the paper

Because this dataset does not directly contain education/income/political covariates in the final analysis tables, the quantitative claim should remain:

1. **Observed result**: topic usage and lexical patterns differ across states.
2. **Interpretation step**: plausible explanatory factors (education, income, political context, industry profile, local culture) are discussed as hypotheses for why differences may emerge.
3. **Causal caution**: these analyses are descriptive and correlational; they do not identify causal drivers.

## 5) Suggested wording for your Results section

“State-level variation is substantial across both topic prevalence and lexical usage. Permutation-based heterogeneity tests show that all BERTopic clusters vary across states beyond chance expectations under a pooled null (all FDR-adjusted q-values < 0.001). Ranking analyses further identify state-specific over-indexing patterns (e.g., Pennsylvania for a sports-script cluster, Massachusetts for a comedic storytelling cluster, Florida for an email/banking-security cluster). Weighted log-odds comparisons (state vs. rest) reproduce these distinctions at the token level, strengthening interpretation consistency across methods.”

## 6) Limitations to include

- Some states have much smaller sample sizes, which can amplify noisier state-vs-rest lexical signals.
- Topic labels are keyword-based and may contain noisy or mixed-intent clusters.
- The current significance test uses a strict pooled-rate null; with large sample size, this can produce uniformly small p-values, so effect sizes/dispersion rankings should be emphasized alongside significance.

## 7) Experiment 2: Spearman with external covariates

Aside from purely geographic concentration analyses, we added an external-covariate test using official ACS 2024 1-year tables for income and education.

### Preprocessing

- Income source: `ACSST1Y2024.S1901-2026-04-20T174538.csv`
- Education source: `ACSST1Y2024.S1501-2026-04-20T174348.csv`
- Script: `src/statistics/preprocess_state_covariates.py`
- Output: `data/state_covariates.csv` (`state`, `income`, `education`)

The preprocessing script reshapes ACS wide-format state columns to a tidy table by extracting:
- median household income (`Households!!Estimate`, row `Median income (dollars)`)
- bachelor's degree or higher for age 25+ (`Percent!!Estimate`, row under `Population 25 years and over`)

### Spearman analysis

- Script: `src/statistics/spearman_with_covariates.py`
- Inputs: `data/state_topic_proportions.parquet` + `data/state_covariates.csv`
- Outputs:
  - `results/statistics/state_topic_with_covariates.csv`
  - `results/statistics/spearman_topic_covariates.csv`
  - `results/statistics/spearman_topic_covariates_significant.csv`

### Key findings (Experiment 2)

- Included states in merged test: **36**
- Total tests: **152** (76 topics x 2 covariates)
- FDR-significant links: **5** total
  - income: **3**
  - education: **2**

Significant associations after FDR:

- **Income**
  - `topic_51` (`detailed, prompt, description...`) rho = 0.571, q = 0.0209
  - `topic_60` (`detailed, description, ar...`) rho = 0.517, q = 0.0323
  - `topic_63` (`scp, dr, researcher, log...`) rho = 0.516, q = 0.0323
- **Education**
  - `topic_63` (`scp, dr, researcher, log...`) rho = 0.533, q = 0.0464
  - `topic_64` (`galaxy, image, markdown...`) rho = -0.518, q = 0.0464

### Notes for interpretation

- Income and education are strongly correlated across included states (rho = 0.905), so overlap in topic associations is expected.
- These are state-level correlational links and should be interpreted as descriptive patterns, not causal effects.
- Run-level snapshot for this experiment is saved at `results/statistics/part_c_results/part_c_result_run3.txt`.

## 8) Additional observations from outcome diagnostics

To avoid over-interpreting isolated coefficients, we ran an additional outcome diagnostic pass:

- Script: `src/statistics/diagnose_spearman_covariates.py`
- Outputs:
  - `results/statistics/part_c_results/part_c_result_run4.txt`
  - `results/statistics/part_c_results/spearman_significant_deep_summary.csv`

### What looks interesting

- The topic-level effect profiles for income and education are highly aligned:
  - correlation between rho vectors = **0.941**
  - same sign in **96.1%** of topics
  - top-ranked overlap is substantial (e.g., top-20 Jaccard = **0.667**)
- This suggests the two covariates are largely tracking a common state-level gradient in this sample.

### Which links look most robust

Bootstrap (4,000 resamples) for FDR-significant links gives non-trivial intervals:

- education ~ `topic_63`: rho = 0.533, 95% CI [0.262, 0.724]
- education ~ `topic_64`: rho = -0.518, 95% CI [-0.768, -0.179]
- income ~ `topic_51`: rho = 0.571, 95% CI [0.307, 0.774]
- income ~ `topic_60`: rho = 0.517, 95% CI [0.228, 0.741]
- income ~ `topic_63`: rho = 0.516, 95% CI [0.261, 0.708]

### Caution points from sparsity/leverage

- Some significant topics are sparse across states (e.g., `topic_63` is non-zero in only 10/36 states), so rank correlations can be sensitive to a few high-prevalence states.
- In contrast, `topic_64` is denser (31/36 non-zero states), making that negative education association less dependent on a tiny subset of states.
- Practical reporting should emphasize both statistical significance and prevalence support (`nonzero_states`) rather than p-values alone.
