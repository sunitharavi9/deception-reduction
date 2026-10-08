# deception-reduction
CAIS Test

## Setup

Two separate environments: a model server (vLLM) and an eval client.

### 1. Model server (Linux + NVIDIA GPU)
```bash
pip install uv
uv venv .venv-vllm --python 3.11
source .venv-vllm/bin/activate
uv pip install -r requirements-vllm.txt --torch-backend=auto
vllm serve Qwen/Qwen3.5-9B --port 8000     # leave running
curl http://localhost:8000/v1/models        # check it's up
```
If vLLM fails with CUDA library errors, point it at the pip-installed CUDA 13 libs first:
```bash
export LD_LIBRARY_PATH=$(python -c "import nvidia,os;print(os.path.dirname(nvidia.__file__))")/cu13/lib:$LD_LIBRARY_PATH
```

On Apple silicon use `mlx_lm.server --model <mlx-qwen-repo> --port 8000` or Ollama instead.

### 2. Eval client
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Data
Benchmarks live in `data/evals/` (not committed): `mask/`, `machiavelli/` (incl. `game_data/`, ~3.4 GB),
`mmlu/`, `gsm8k/`, `mt_bench/`. Build the small dev subsets with:
```bash
python scripts/make_dev_sets.py
```

### 4. Run evals
```bash
# smoke test
python scripts/run_evals.py --split dev --model Qwen/Qwen3.5-9B --bench mmlu gsm8k mt_bench --limit 5

# full dev run (all five benchmarks)
python scripts/run_evals.py --split dev --model Qwen/Qwen3.5-9B

# full test split, MASK only
python scripts/run_evals.py --split test --model Qwen/Qwen3.5-9B --bench mask
```
Benchmarks: `mask`, `machiavelli`, `mmlu`, `gsm8k`, `mt_bench`. Results go to
`outputs/<model>_<split>/summary.md`; reruns resume from cached rows. If the server is on another
machine, add `--base-url http://<host>:8000/v1`. MASK and MT-Bench are scored by an LLM judge that
defaults to the model under test; pass `--judge-model` / `--judge-base-url` for a stronger judge.

### Running the model and a separate judge together (8 GPUs, no quantization)
Give each server its own GPUs so they don't share memory. The 9B model (~18 GB in bf16) fits on one GPU;
the 32B judge (~64 GB in bf16) is split over two GPUs with tensor parallelism. The remaining GPUs are free.
Start the model first and wait until it is up, then start the judge (separate terminals).
```bash
# 1) model under test on GPU 0
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-9B --port 8000 \
  --gpu-memory-utilization 0.90 --max-model-len 16384

# 2) judge on GPUs 1-2 (bf16, tensor parallel 2)
CUDA_VISIBLE_DEVICES=1,2 vllm serve Qwen/Qwen2.5-32B-Instruct --port 8001 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 --max-model-len 16384

# 3) run evals against both
python scripts/run_evals.py --split test --model Qwen/Qwen3.5-9B \
  --base-url http://localhost:8000/v1 \
  --judge-model Qwen/Qwen2.5-32B-Instruct --judge-base-url http://localhost:8001/v1 \
  --concurrency 32 --out outputs/baseline_full_test
```
For more throughput, add `--data-parallel-size N` to the model server (e.g. `CUDA_VISIBLE_DEVICES=0,1,2,3` and
`--data-parallel-size 4`, moving the judge to GPUs 4-5).
