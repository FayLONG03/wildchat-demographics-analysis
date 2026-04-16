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

---

## Section 3: Word Statistics

---

## Section 4: Qualitative Analysis
