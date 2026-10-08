#!/usr/bin/env bash
# Rebuild both envs on a fresh machine.  Usage: DATA_ROOT=/big bash scripts/setup_env.sh
# DATA_ROOT should be a volume with >=100 GB free; caches and venvs go there.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT}"
export UV_CACHE_DIR="$DATA_ROOT/.uv-cache" HF_HOME="$DATA_ROOT/.hf-cache" TMPDIR="$DATA_ROOT/.tmp"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TMPDIR"
pip install -q -U pip uv

# model server (vLLM)
python3 -m venv "$DATA_ROOT/.venv-vllm"
"$DATA_ROOT/.venv-vllm/bin/pip" install -q -U pip uv
(. "$DATA_ROOT/.venv-vllm/bin/activate" && uv pip install -r "$ROOT/requirements-vllm.txt" --torch-backend=auto)

# eval client (needs Python <= 3.11 for MACHIAVELLI's gym/spacy pins)
python3 -m venv "$DATA_ROOT/.venv"
"$DATA_ROOT/.venv/bin/pip" install -q -r "$ROOT/requirements.txt"

cat <<MSG
Done. Add to ~/.bashrc:
  export HF_HOME=$HF_HOME UV_CACHE_DIR=$UV_CACHE_DIR
Server:  source $DATA_ROOT/.venv-vllm/bin/activate && vllm serve Qwen/Qwen3.5-9B --port 8000
Evals:   source $DATA_ROOT/.venv/bin/activate && python scripts/run_evals.py --split dev
MACHIAVELLI game_data (3.4 GB) is gitignored: rsync data/evals/machiavelli/game_data or download it per its README.
MSG
