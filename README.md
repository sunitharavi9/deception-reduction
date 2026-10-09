# deception-reduction

## Setup

```bash
# vLLM env (model + judge servers)
pip install uv
uv venv .venv-vllm --python 3.11
source .venv-vllm/bin/activate
uv pip install -r requirements-vllm.txt --torch-backend=auto

# eval client env
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# training env (LoRRA, SFT/DPO, steering)
uv venv .venv-train --python 3.11
source .venv-train/bin/activate
uv pip install torch --torch-backend=cu128
uv pip install -r requirements-train.txt
```

Machiavelli game data (gitignored):
```bash
cd data/evals/machiavelli
source ../../../.venv/bin/activate
pip install gdown
gdown "https://drive.google.com/uc?id=19PXa2bgjkfFfTTI3EZIT3-IJ_vxrV0Rz" -O game_data.zip
unzip -q -P machiavelli game_data.zip && rm game_data.zip
ls game_data/game_metadata.json
cd ../../..
python scripts/make_dev_sets.py        # dev subsets (data/evals/dev/)
```

## Baseline evals

```bash
# model under test: 4 replicas on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3.5-9B --port 8000 \
  --data-parallel-size 4 --gpu-memory-utilization 0.90 --max-model-len 16384

# judge on GPUs 4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 --port 8001 \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.90 --max-model-len 16384

# smoke test
python scripts/run_evals.py --split dev --limit 5 --model Qwen/Qwen3.5-9B \
  --base-url http://localhost:8000/v1 \
  --judge-model mistralai/Mistral-Small-3.2-24B-Instruct-2506 --judge-base-url http://localhost:8001/v1

# full test split, all five benchmarks
python scripts/run_evals.py --split test --model Qwen/Qwen3.5-9B \
  --base-url http://localhost:8000/v1 \
  --judge-model mistralai/Mistral-Small-3.2-24B-Instruct-2506 --judge-base-url http://localhost:8001/v1 \
  --concurrency 64 --mach-workers 30 \
  --out outputs/baseline_full_test

# re-judge cached generations (e.g. after changing the judge)
python scripts/run_evals.py --split test --bench mask --model Qwen/Qwen3.5-9B --rejudge \
  --judge-model mistralai/Mistral-Small-3.2-24B-Instruct-2506 --judge-base-url http://localhost:8001/v1 \
  --out outputs/baseline_full_test
```

Benchmarks: `--bench mask machiavelli mmlu gsm8k mt_bench` (default: all). Other flags: `python scripts/run_evals.py --help`.

## Experiment 1: developer system prompts

```bash
for V in honest honest_ethical; do
  python scripts/run_evals.py --split test --model Qwen/Qwen3.5-9B --developer-prompt $V \
    --base-url http://localhost:8000/v1 \
    --judge-model mistralai/Mistral-Small-3.2-24B-Instruct-2506 --judge-base-url http://localhost:8001/v1 \
    --concurrency 64 --mach-workers 30 --out outputs/devprompt_${V}_test
done
```

Prompt text: `DEVELOPER_PROMPTS` in `scripts/run_evals.py`. Use a new `--out` per variant.

## Experiment 2: LoRRA

```bash
# train (stop vLLM first) -> models/qwen3.5-9b-lorra
source .venv-train/bin/activate
torchrun --standalone --nproc_per_node=8 scripts/train_lorra.py

# serve (GPUs 0-5) and evaluate; judge on GPUs 6-7
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vllm serve models/qwen3.5-9b-lorra --served-model-name qwen3.5-9b-lorra \
  --port 8000 --data-parallel-size 6 --gpu-memory-utilization 0.90 --max-model-len 16384
CUDA_VISIBLE_DEVICES=6,7 vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 --port 8001 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 --max-model-len 16384

python scripts/run_evals.py --split test --model qwen3.5-9b-lorra --base-url http://localhost:8000/v1 \
  --judge-model mistralai/Mistral-Small-3.2-24B-Instruct-2506 --judge-base-url http://localhost:8001/v1 \
  --concurrency 96 --mach-workers 30 --out outputs/lorra_full_test
```

Options: `python scripts/train_lorra.py --help`.

## Experiment 3: SFT / DPO on honesty + ethics data

```bash
# 1) synthetic scenarios (generator model must not be the eval judge)
CUDA_VISIBLE_DEVICES=6,7 vllm serve Qwen/Qwen2.5-32B-Instruct --port 8002 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 --max-model-len 16384
python scripts/gen_synthetic_honesty.py --n 3000 --concurrency 64 \
  --teacher-model Qwen/Qwen2.5-32B-Instruct --teacher-base-url http://localhost:8002/v1

# 2) check DolusChat's fields, then build the three datasets (use the same --tulu-exclude flags for all)
python scripts/build_training_data.py --inspect doluschat
python scripts/build_training_data.py                                               # data/train
python scripts/build_training_data.py --tulu-only                                   # data/train/control_tulu
python scripts/build_training_data.py --synthetic '' --out data/train/no_synthetic  # without synthetic data
cat data/train/manifest.json; less data/train/samples.md

# 3) trial run, then all five training runs (stop vLLM first; finished runs are skipped)
source .venv-train/bin/activate
torchrun --standalone --nproc_per_node=8 scripts/train_sft_dpo.py --stage sft --max-examples 512 --out-dir models/sft_trial
bash scripts/run_training_suite.sh

# 4) evaluate all five (MASK judge on :8001, GPUs 6-7)
bash scripts/eval_models.sh qwen3.5-9b-sft qwen3.5-9b-sft-tulu-only qwen3.5-9b-sft-nosyn \
  qwen3.5-9b-sft-dpo qwen3.5-9b-sft-nosyn-dpo
```

DPO without an SFT stage:
```bash
torchrun --standalone --nproc_per_node=8 scripts/train_sft_dpo.py --stage dpo --model Qwen/Qwen3.5-9B \
  --rpo-alpha 0.2 --out-dir models/qwen3.5-9b-dpo-only
```

Options: `python scripts/build_training_data.py --help`, `python scripts/train_sft_dpo.py --help`.

## Future work:  activation steering

```bash
source .venv-train/bin/activate

# 1) vectors for all layers (one GPU); note heldout_cosine in models/steer/qwen3.5-9b/meta.json
CUDA_VISIBLE_DEVICES=0 python scripts/steer_honesty.py compute --model Qwen/Qwen3.5-9B --n 1000 \
  --out models/steer/qwen3.5-9b

# 2) one steer server per GPU 0-5
for i in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$i nohup python scripts/steer_honesty.py serve --model Qwen/Qwen3.5-9B \
    --vectors models/steer/qwen3.5-9b/vectors.pt --port 900$i > steer_$i.log 2>&1 &
done

# 3) layer x alpha sweep on capability dev sets (MASK is held out)
python scripts/steer_sweep.py --ports 9000 9001 9002 9003 9004 9005 \
  --layers <layers> --alphas 0.5 1 2 4

# 4) final: MASK once on the chosen setting (judge on :8001)
curl -X POST localhost:9000/steer -d '{"layers":[<L>],"alpha":<A>}'
python scripts/run_evals.py --split test --bench mask --model qwen3.5-9b-steer --base-url http://localhost:9000/v1 \
  --judge-model mistralai/Mistral-Small-3.2-24B-Instruct-2506 --judge-base-url http://localhost:8001/v1 \
  --concurrency 32 --out outputs/steer_final_test
```

## Analysis: LoRRA vs base activations

```bash
# needs the steering vectors from Experiment 4, step 1
CUDA_VISIBLE_DEVICES=0 python scripts/analyze_lorra.py \
  --adapter models/qwen3.5-9b-lorra/adapter --vectors models/steer/qwen3.5-9b/vectors.pt --weights
# -> outputs/lorra_analysis/{summary.md,results.json,layers_main.png,layers_last.png}
```
