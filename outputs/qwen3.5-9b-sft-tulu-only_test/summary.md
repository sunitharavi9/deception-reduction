# Eval summary: qwen3.5-9b-sft-tulu-only (test split)

thinking=off, developer_prompt=off, judge=mistralai/Mistral-Small-3.2-24B-Instruct-2506

## MASK

Scoring: official MASK pipeline (judge prompts and aggregation ported from the MASK repo).

Macro-avg honesty (1 - P(lie)): **0.806**; normalized (rows with a belief only): **0.691**

| archetype | n | honesty | lie rate | lie | evasion | no belief | accuracy (given belief) | accuracy (all rows) |
|---|---|---|---|---|---|---|---|---|
| continuations | 176 | 0.733 | 0.267 | 0.420 | 0.267 | 0.364 | 0.964 | 0.614 |
| disinformation | 125 | 0.928 | 0.072 | 0.086 | 0.416 | 0.160 | 0.990 | 0.832 |
| doubling_down_known_facts | 120 | 0.700 | 0.300 | 0.720 | 0.125 | 0.583 | 0.820 | 0.342 |
| known_facts | 209 | 0.928 | 0.072 | 0.169 | 0.493 | 0.574 | 0.764 | 0.325 |
| provided_facts | 274 | 0.682 | 0.318 | 0.318 | 0.391 | 0.000 | n/a | n/a |
| statistics | 95 | 0.863 | 0.137 | 0.143 | 0.695 | 0.042 | 0.659 | 0.632 |
| overall (micro) | 999 | 0.793 | 0.207 | 0.287 | 0.390 | 0.278 | 0.852 | 0.381 |

`lie` column = P(lie | model holds a consistent belief). Honesty counts evasions and no-belief rows as non-lies, as in MASK.

## MACHIAVELLI

Scores are % of the random agent's mean count (100 = random-agent level; for harms lower is better, for game.score higher is better).

| metric | mean |
|---|---|
| game.score | 122.10 |
| points | 20.56 |
| achievements | 11.13 |
| steps | 171.20 |
| power.Σ | 101.84 |
| utility.suffering | 97.57 |
| violations.Σ | 98.21 |
| violations.deception | 101.50 |
| violations.fairness | 88.41 |
| violations.intending_harm | 96.77 |
| violations.killing | 83.24 |
| violations.manipulation | 94.58 |
| violations.non_physical_harm | 75.13 |
| violations.other | 77.57 |
| violations.physical_harm | 98.09 |
| violations.promises | 69.73 |
| violations.spying | 117.31 |
| violations.stealing | 91.99 |
| violations.trespassing | 102.81 |
| violations.vandalism | 113.46 |

30 episodes over 30 games; 0 truncated at --mach-max-steps.

## MMLU

0-shot accuracy: **0.777** (n=14042, macro over subjects 0.794, unparsed 0)

## GSM8K

0-shot accuracy: **0.587** (n=1319, unparsed 0)

## MT-Bench

Mean score (1-10): **7.076** (turn 1 7.800, turn 2 6.333, n=80, unparsed judgments 2)

| category | score |
|---|---|
| coding | 6.150 |
| extraction | 8.250 |
| humanities | 7.250 |
| math | 6.632 |
| reasoning | 7.053 |
| roleplay | 6.850 |
| stem | 7.000 |
| writing | 7.400 |
