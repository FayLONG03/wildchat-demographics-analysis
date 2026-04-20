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

We ran weighted log-odds on cleaned user text (`wildchat_log_odds.parquet`) for each state against all remaining states, using an informative prior and minimum token-frequency threshold. Tokenization used lowercase regex filtering with stopword removal and minimum token length constraints to improve interpretability. States with at least 100 prompts were included (39 states; state prompt counts ranged from 103 to 11,254).

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
