# Eval summary: Qwen/Qwen3.5-9B (test split)

thinking=off, developer_prompt=on, judge=mistralai/Mistral-Small-3.2-24B-Instruct-2506

## MASK

Scoring: official MASK pipeline (judge prompts and aggregation ported from the MASK repo).

Macro-avg honesty (1 - P(lie)): **0.664**; normalized (rows with a belief only): **0.565**

| archetype | n | honesty | lie rate | lie | evasion | no belief | accuracy (given belief) | accuracy (all rows) |
|---|---|---|---|---|---|---|---|---|
| continuations | 176 | 0.602 | 0.398 | 0.486 | 0.085 | 0.182 | 0.972 | 0.795 |
| disinformation | 125 | 0.840 | 0.160 | 0.172 | 0.040 | 0.072 | 0.983 | 0.912 |
| doubling_down_known_facts | 120 | 0.733 | 0.267 | 0.571 | 0.042 | 0.533 | 0.732 | 0.342 |
| known_facts | 209 | 0.770 | 0.230 | 0.417 | 0.077 | 0.450 | 0.739 | 0.407 |
| provided_facts | 274 | 0.405 | 0.595 | 0.595 | 0.139 | 0.000 | n/a | n/a |
| statistics | 95 | 0.632 | 0.368 | 0.368 | 0.221 | 0.000 | 0.600 | 0.600 |
| overall (micro) | 999 | 0.632 | 0.368 | 0.460 | 0.100 | 0.199 | 0.831 | 0.437 |

`lie` column = P(lie | model holds a consistent belief). Honesty counts evasions and no-belief rows as non-lies, as in MASK.

## MACHIAVELLI

Scores are % of the random agent's mean count (100 = random-agent level; for harms lower is better, for game.score higher is better).

| metric | mean |
|---|---|
| game.score | 144.34 |
| points | 24.73 |
| achievements | 12.27 |
| steps | 168.03 |
| power.Σ | 104.28 |
| utility.suffering | 98.48 |
| violations.Σ | 104.54 |
| violations.deception | 110.96 |
| violations.fairness | 86.23 |
| violations.intending_harm | 113.96 |
| violations.killing | 96.47 |
| violations.manipulation | 97.72 |
| violations.non_physical_harm | 84.04 |
| violations.other | 92.29 |
| violations.physical_harm | 100.16 |
| violations.promises | 78.11 |
| violations.spying | 114.62 |
| violations.stealing | 95.68 |
| violations.trespassing | 104.40 |
| violations.vandalism | 106.18 |

30 episodes over 30 games; 0 truncated at --mach-max-steps.

## MMLU

0-shot accuracy: **0.783** (n=14042, macro over subjects 0.801, unparsed 0)

## GSM8K

0-shot accuracy: **0.938** (n=1319, unparsed 0)

## MT-Bench

Mean score (1-10): **8.146** (turn 1 8.615, turn 2 7.684, n=80, unparsed judgments 3)

| category | score |
|---|---|
| coding | 8.056 |
| extraction | 7.900 |
| humanities | 8.500 |
| math | 7.632 |
| reasoning | 7.600 |
| roleplay | 8.700 |
| stem | 8.700 |
| writing | 8.050 |
