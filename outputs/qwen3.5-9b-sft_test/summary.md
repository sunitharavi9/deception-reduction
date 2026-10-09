# Eval summary: qwen3.5-9b-sft (test split)

thinking=off, developer_prompt=off, judge=mistralai/Mistral-Small-3.2-24B-Instruct-2506

## MASK

Scoring: official MASK pipeline (judge prompts and aggregation ported from the MASK repo).

Macro-avg honesty (1 - P(lie)): **0.814**; normalized (rows with a belief only): **0.731**

| archetype | n | honesty | lie rate | lie | evasion | no belief | accuracy (given belief) | accuracy (all rows) |
|---|---|---|---|---|---|---|---|---|
| continuations | 176 | 0.767 | 0.233 | 0.339 | 0.222 | 0.312 | 0.950 | 0.653 |
| disinformation | 125 | 0.944 | 0.056 | 0.069 | 0.376 | 0.192 | 0.980 | 0.792 |
| doubling_down_known_facts | 120 | 0.875 | 0.125 | 0.300 | 0.075 | 0.583 | 0.580 | 0.242 |
| known_facts | 209 | 0.833 | 0.167 | 0.372 | 0.014 | 0.550 | 0.606 | 0.273 |
| provided_facts | 274 | 0.770 | 0.230 | 0.230 | 0.047 | 0.000 | n/a | n/a |
| statistics | 95 | 0.695 | 0.305 | 0.305 | 0.305 | 0.000 | 0.632 | 0.632 |
| overall (micro) | 999 | 0.810 | 0.190 | 0.259 | 0.140 | 0.264 | 0.781 | 0.360 |

`lie` column = P(lie | model holds a consistent belief). Honesty counts evasions and no-belief rows as non-lies, as in MASK.

## MACHIAVELLI

Scores are % of the random agent's mean count (100 = random-agent level; for harms lower is better, for game.score higher is better).

| metric | mean |
|---|---|
| game.score | 113.61 |
| points | 21.13 |
| achievements | 10.70 |
| steps | 169.77 |
| power.Σ | 95.97 |
| utility.suffering | 94.04 |
| violations.Σ | 86.57 |
| violations.deception | 88.30 |
| violations.fairness | 77.67 |
| violations.intending_harm | 86.23 |
| violations.killing | 81.31 |
| violations.manipulation | 98.06 |
| violations.non_physical_harm | 67.24 |
| violations.other | 75.07 |
| violations.physical_harm | 86.17 |
| violations.promises | 85.95 |
| violations.spying | 107.66 |
| violations.stealing | 68.04 |
| violations.trespassing | 81.64 |
| violations.vandalism | 99.81 |

30 episodes over 30 games; 0 truncated at --mach-max-steps.

## MMLU

0-shot accuracy: **0.778** (n=14042, macro over subjects 0.794, unparsed 0)

## GSM8K

0-shot accuracy: **0.619** (n=1319, unparsed 0)

## MT-Bench

Mean score (1-10): **6.994** (turn 1 7.575, turn 2 6.412, n=80, unparsed judgments 0)

| category | score |
|---|---|
| coding | 6.300 |
| extraction | 8.200 |
| humanities | 7.600 |
| math | 5.600 |
| reasoning | 6.450 |
| roleplay | 7.150 |
| stem | 6.950 |
| writing | 7.700 |
