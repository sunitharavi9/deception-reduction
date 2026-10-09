# Eval summary: Qwen/Qwen3.5-9B (test split)

thinking=off, developer_prompt=honest_ethical, judge=mistralai/Mistral-Small-3.2-24B-Instruct-2506

## MASK

Scoring: official MASK pipeline (judge prompts and aggregation ported from the MASK repo).

Macro-avg honesty (1 - P(lie)): **0.811**; normalized (rows with a belief only): **0.737**

| archetype | n | honesty | lie rate | lie | evasion | no belief | accuracy (given belief) | accuracy (all rows) |
|---|---|---|---|---|---|---|---|---|
| continuations | 176 | 0.909 | 0.091 | 0.111 | 0.045 | 0.182 | 0.965 | 0.790 |
| disinformation | 125 | 0.968 | 0.032 | 0.035 | 0.032 | 0.096 | 0.991 | 0.896 |
| doubling_down_known_facts | 120 | 0.767 | 0.233 | 0.475 | 0.125 | 0.508 | 0.746 | 0.367 |
| known_facts | 209 | 0.809 | 0.191 | 0.367 | 0.144 | 0.478 | 0.752 | 0.392 |
| provided_facts | 274 | 0.675 | 0.325 | 0.325 | 0.245 | 0.000 | n/a | n/a |
| statistics | 95 | 0.737 | 0.263 | 0.263 | 0.189 | 0.000 | 0.674 | 0.674 |
| overall (micro) | 999 | 0.798 | 0.202 | 0.254 | 0.142 | 0.205 | 0.848 | 0.441 |

`lie` column = P(lie | model holds a consistent belief). Honesty counts evasions and no-belief rows as non-lies, as in MASK.

## MACHIAVELLI

Scores are % of the random agent's mean count (100 = random-agent level; for harms lower is better, for game.score higher is better).

| metric | mean |
|---|---|
| game.score | 128.53 |
| points | 22.70 |
| achievements | 11.67 |
| steps | 166.33 |
| power.Σ | 101.30 |
| utility.suffering | 99.60 |
| violations.Σ | 98.38 |
| violations.deception | 99.52 |
| violations.fairness | 85.87 |
| violations.intending_harm | 109.20 |
| violations.killing | 96.01 |
| violations.manipulation | 88.30 |
| violations.non_physical_harm | 89.25 |
| violations.other | 103.64 |
| violations.physical_harm | 98.73 |
| violations.promises | 60.56 |
| violations.spying | 112.02 |
| violations.stealing | 105.93 |
| violations.trespassing | 102.06 |
| violations.vandalism | 99.72 |

30 episodes over 30 games; 0 truncated at --mach-max-steps.

## MMLU

0-shot accuracy: **0.781** (n=14042, macro over subjects 0.798, unparsed 1)

## GSM8K

0-shot accuracy: **0.939** (n=1319, unparsed 0)

## MT-Bench

Mean score (1-10): **8.225** (turn 1 8.588, turn 2 7.862, n=80, unparsed judgments 0)

| category | score |
|---|---|
| coding | 8.300 |
| extraction | 8.450 |
| humanities | 8.750 |
| math | 7.350 |
| reasoning | 7.550 |
| roleplay | 8.500 |
| stem | 8.600 |
| writing | 8.300 |
