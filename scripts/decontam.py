"""N-gram decontamination against the eval sets in data/evals/ (MASK, MT-Bench, GSM8K test, MMLU test).

A training text is flagged when it shares any word n-gram (default 13, the usual LM-decontamination setting)
with an eval prompt, or when it contains a benchmark canary string. Eval texts too short to form a 13-gram
(29% of MMLU test questions) would never match that way, so eval texts of at least `min_exact` words (default 8)
but fewer than 13 are matched as exact contiguous word sequences instead. Eval texts under `min_exact` words are
not checked (a match would be mostly a false positive). MACHIAVELLI is not indexed: its game
text lives in data/evals/machiavelli/game_data (3.4 GB, gitignored) and none of our sources are drawn from it.

    from decontam import EvalIndex
    idx = EvalIndex.build()          # ~15 s, mostly MMLU
    idx.hit("some training text")    # -> name of the matching eval set, or None
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "data" / "evals"
CANARY_RE = re.compile(r"canary GUID|BENCHMARK DATA SHOULD NEVER APPEAR", re.I)
WORD_RE = re.compile(r"[a-z0-9]+")
MASK_TEXT_COLS = ["system_prompt", "user_prompt", "dd_turn_1_prompt", "dd_turn_1_response", "proposition",
                  "ground_truth", "belief_elicit_1", "belief_elicit_2", "belief_elicit_3"]


def words(text):
    return WORD_RE.findall(str(text).lower())


def ngrams(text, n):
    w = words(text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def mask_frame():
    """All MASK test rows (every archetype) with an `archetype` column."""
    frames = []
    for d in sorted((EVALS / "mask").iterdir()):
        if d.is_dir():
            df = pd.read_parquet(d / "test-00000-of-00001.parquet")
            frames.append(df.assign(archetype=d.name))
    return pd.concat(frames, ignore_index=True)


def eval_texts():
    """(eval name, text) for every eval prompt we must not train on."""
    m = mask_frame()
    for c in MASK_TEXT_COLS:
        if c in m:
            for t in m[c].dropna():
                yield "mask", t
    for line in (EVALS / "mt_bench" / "question.jsonl").read_text().splitlines():
        for t in json.loads(line)["turns"]:
            yield "mt_bench", t
    gsm = EVALS / "gsm8k" / "test.parquet"
    if gsm.exists():
        for t in pd.read_parquet(gsm)["question"]:
            yield "gsm8k", t
    mmlu = EVALS / "mmlu" / "test.parquet"
    if mmlu.exists():
        for t in pd.read_parquet(mmlu)["question"]:
            yield "mmlu", t


class EvalIndex:
    def __init__(self, grams, n, short):
        self.grams, self.n = grams, n
        self.short = short  # {num_words: {normalized text: eval name}} for eval texts shorter than n words
        self.starts = {t.split(" ", 1)[0] for d in short.values() for t in d}  # cheap pre-filter

    @classmethod
    def build(cls, n=13, min_exact=8):
        grams, short = {}, {}
        for name, text in eval_texts():
            w = words(text)
            if len(w) >= n:
                for g in ngrams(text, n):
                    grams.setdefault(g, name)
            elif len(w) >= min_exact:
                short.setdefault(len(w), {}).setdefault(" ".join(w), name)
        return cls(grams, n, short)

    def hit(self, text):
        if CANARY_RE.search(text):
            return "canary"
        for g in ngrams(text, self.n):
            if g in self.grams:
                return self.grams[g]
        if self.short:
            w = words(text)
            for i, first in enumerate(w):
                if first in self.starts:
                    for length, table in self.short.items():
                        if i + length <= len(w):
                            name = table.get(" ".join(w[i:i + length]))
                            if name:
                                return name
        return None


def example_text(ex):
    """All text in an SFT ({messages}) or DPO ({prompt, chosen, rejected}) example."""
    parts = []
    for k in ("messages", "prompt", "chosen", "rejected"):
        for m in ex.get(k) or []:
            parts.append(m["content"])
    return "\n".join(parts)


class MaskTopicFilter:
    """Looser check for synthetic data: drop a generated fact whose content words overlap a MASK proposition or
    ground truth too much (Jaccard), so synthetic scenarios stay off MASK's topics, not just its exact wording."""

    STOP = set("a an the of to in on for and or is are was were be been it its this that with as by at from "
               "than more less not no do does did has have had will would can could should".split())

    def __init__(self, threshold=0.34):
        self.threshold = threshold
        m = mask_frame()
        texts = list(m["proposition"].dropna()) + list(m["ground_truth"].dropna())
        self.sets = [s for s in (self._content(t) for t in texts) if len(s) >= 3]

    def _content(self, text):
        text = re.sub(r"</?pivotal>|<value>", " ", str(text))
        return {w for w in words(text) if w not in self.STOP and len(w) > 2}

    def max_overlap(self, text):
        s = self._content(text)
        if not s:
            return 0.0
        return max((len(s & t) / len(s | t) for t in self.sets), default=0.0)

    def too_close(self, text):
        return self.max_overlap(text) >= self.threshold
