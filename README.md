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
`mmlu/`, `gsm8k/`, `mt_bench/`. The Machiavelli game data is gitignored, so on a new machine download it:
```bash
cd deception-reduction/data/evals/machiavelli
source ../../../.venv/bin/activate
pip install gdown
gdown "https://drive.google.com/uc?id=19PXa2bgjkfFfTTI3EZIT3-IJ_vxrV0Rz" -O game_data.zip
unzip -q -P machiavelli game_data.zip && rm game_data.zip
ls game_data/game_metadata.json     # should exist
```
Build the small dev subsets with:
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

### Running the model and a separate judge together (all 8 GPUs, no quantization)
MASK is scored with the official MASK pipeline (judge prompts and belief/honesty aggregation from
`centerforaisafety/mask`, see `scripts/mask_official.py`). The judge should be a different, strong model from
the one under test; we use a non-Qwen judge so it is not the same family as the model being evaluated.

Layout: the 9B model is replicated on GPUs 0-3 (data parallel); the judge, Llama-3.3-70B-Instruct (~140 GB in
bf16), is split over GPUs 4-7 (tensor parallel). Llama is gated: accept the license on Hugging Face and
`export HF_TOKEN=...` first. Start the model first and wait until it is up, then the judge (separate terminals).
```bash
# 1) model under test: 4 replicas on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3.5-9B --port 8000 \
  --data-parallel-size 4 --gpu-memory-utilization 0.90 --max-model-len 16384

# 2) judge on GPUs 4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8001 \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.90 --max-model-len 16384

# 3) baseline evals: all five benchmarks, full test split
python scripts/run_evals.py --split test --model Qwen/Qwen3.5-9B \
  --base-url http://localhost:8000/v1 \
  --judge-model meta-llama/Llama-3.3-70B-Instruct --judge-base-url http://localhost:8001/v1 \
  --concurrency 64 --mach-workers 30 \
  --out outputs/baseline_full_test
```
No gated access? Use an ungated non-Qwen judge instead, e.g. `mistralai/Mistral-Small-3.2-24B-Instruct-2506`
(~48 GB) on two GPUs: `CUDA_VISIBLE_DEVICES=6,7 vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 --port 8001 --tensor-parallel-size 2 ...`
and give the model GPUs 0-5 with `--data-parallel-size 6`.

`--concurrency` is the number of parallel MASK / MMLU / GSM8K / MT-Bench requests; `--mach-workers` is the
number of Machiavelli games played in parallel (one process per game, 30 test games). Lower either if the
judge becomes the bottleneck. Reruns with the same `--out` resume from cached rows.

**Re-judging existing generations.** Rows in `--out` generated earlier (or scored with a different judge) keep
their generations; only the judge step is repeated. To rescore a finished run with a new judge, you only need
the judge server up:
```bash
python scripts/run_evals.py --split test --bench mask --model Qwen/Qwen3.5-9B \
  --judge-model meta-llama/Llama-3.3-70B-Instruct --judge-base-url http://localhost:8001/v1 \
  --out outputs/baseline_full_test
```
Rows already scored are skipped unless you pass `--rejudge`, which re-runs the judge on every cached
generation (use it when switching judge models; copy the old folder to a new `--out` first if you want to keep
both sets of scores).

### Experiment 1: honesty "developer system prompt" (MASK paper §5.2, App. A.6)
Same command as the baseline, plus `--developer-prompt` (prepended to MASK pressure prompts only; belief
elicitation is unchanged). Use a new `--out` so it doesn't resume from baseline rows.
```bash
python scripts/run_evals.py --split test --model Qwen/Qwen3.5-9B --bench mask --developer-prompt \
  --base-url http://localhost:8000/v1 --judge-model meta-llama/Llama-3.3-70B-Instruct \
  --judge-base-url http://localhost:8001/v1 --concurrency 64 --out outputs/devprompt_mask_test
```

### Experiment 2: LoRRA
Trains LoRRA adapters on Qwen3.5-9B (MASK paper §5.2) and merges them into a vLLM-loadable checkpoint.
`scripts/train_lorra.py` is data-parallel over all GPUs (the notebook `notebooks/lorra_honesty.ipynb` is the
same recipe on a single GPU). Stop any vLLM servers first, since training takes the GPUs.

```bash
# one-time: separate training env (torch matched to the driver, e.g. CUDA 12.8)
uv venv .venv-train --python 3.11 && source .venv-train/bin/activate
uv pip install torch --torch-backend=cu128
uv pip install -r requirements-train.txt

# short trial first (~160 examples) to check it runs and read the time per step
torchrun --standalone --nproc_per_node=8 scripts/train_lorra.py --n-train 160 --out-dir models/lorra_trial

# full run: 8 GPUs, 5000 examples, effective batch 16 (~312 optimizer steps) -> models/qwen3.5-9b-lorra
torchrun --standalone --nproc_per_node=8 scripts/train_lorra.py
```
Use `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 ...` for a subset of GPUs. The global batch
must be divisible by `--micro-batch` x number of GPUs (defaults: 16 and 2, so 1, 2, 4 or 8 GPUs). Rank 0 prints the loss
every 10 steps, then a held-out sanity check (the cosine should be clearly positive), and saves the merged model plus
`adapter/`, `loss_log.json` and `lorra_config.json` in `--out-dir`. Config choices (alpha, layers, rank, data) are in
`python scripts/train_lorra.py --help`; the paper does not specify them.

Evaluate the merged model with the same judge as the baseline:
```bash
# model under test on GPUs 0-5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vllm serve models/qwen3.5-9b-lorra --served-model-name qwen3.5-9b-lorra \
  --port 8000 --data-parallel-size 6 --gpu-memory-utilization 0.90 --max-model-len 16384
# judge on GPUs 6-7 (see "Running the model and a separate judge together" for the judge command)

python scripts/run_evals.py --split test --model qwen3.5-9b-lorra --base-url http://localhost:8000/v1 \
  --judge-model mistralai/Mistral-Small-3.2-24B-Instruct-2506 --judge-base-url http://localhost:8001/v1 \
  --concurrency 96 --mach-workers 30 --out outputs/lorra_full_test
```
Smoke test first with `--split dev --limit 5 --bench mask mmlu`.
