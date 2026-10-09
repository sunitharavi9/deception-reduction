"""Build small dev subsets of MASK, MACHIAVELLI, MMLU, GSM8K and MT-Bench under data/evals/dev/."""
import json
from pathlib import Path

import pandas as pd

EVALS = Path(__file__).resolve().parents[2] / "data" / "evals"
DEV = EVALS / "dev"
SEED = 0
MASK_PER_SPLIT = 25
MMLU_PER_SUBJECT = 2  # 57 subjects -> 114 questions
GSM8K_N = 100
MTBENCH_PER_CATEGORY = 3  # 8 categories -> 24 questions
# 5 smallest games of the official 30-game MACHIAVELLI test set (by annotation size)
MACHIAVELLI_DEV_GAMES = [
    "ratings-war",
    "hero-project-redemption-season",
    "versus-the-lost-ones",
    "kung-fu",
    "battlemage",
]


def make_mask():
    out = DEV / "mask"
    out.mkdir(parents=True, exist_ok=True)
    for split_dir in sorted((EVALS / "mask").iterdir()):
        if not split_dir.is_dir():
            continue
        df = pd.read_parquet(split_dir / "test-00000-of-00001.parquet")
        dev = df.sample(n=min(MASK_PER_SPLIT, len(df)), random_state=SEED).sort_index()
        dev.to_parquet(out / f"{split_dir.name}.parquet")
        print(f"mask/{split_dir.name}: {len(dev)}/{len(df)}")


def make_machiavelli():
    out = DEV / "machiavelli"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "game_data_dir": "data/evals/machiavelli/game_data",
        "games": MACHIAVELLI_DEV_GAMES,
    }
    (out / "dev_games.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("machiavelli games:", ", ".join(MACHIAVELLI_DEV_GAMES))


def make_mmlu():
    DEV.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(EVALS / "mmlu" / "test.parquet")
    dev = df.groupby("subject", group_keys=False).sample(n=MMLU_PER_SUBJECT, random_state=SEED)
    dev.sort_index().to_parquet(DEV / "mmlu.parquet")
    print(f"mmlu: {len(dev)}/{len(df)}")


def make_gsm8k():
    DEV.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(EVALS / "gsm8k" / "test.parquet")
    dev = df.sample(n=GSM8K_N, random_state=SEED).sort_index()
    dev.to_parquet(DEV / "gsm8k.parquet")
    print(f"gsm8k: {len(dev)}/{len(df)}")


def make_mt_bench():
    out = DEV / "mt_bench"
    out.mkdir(parents=True, exist_ok=True)
    qs = pd.read_json(EVALS / "mt_bench" / "question.jsonl", lines=True)
    dev = qs.groupby("category", group_keys=False).sample(n=MTBENCH_PER_CATEGORY, random_state=SEED)
    dev = dev.sort_values("question_id")
    dev.to_json(out / "question.jsonl", orient="records", lines=True, force_ascii=False)
    print(f"mt_bench: {len(dev)}/{len(qs)}")


if __name__ == "__main__":
    make_mask()
    make_machiavelli()
    make_mmlu()
    make_gsm8k()
    make_mt_bench()
