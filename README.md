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
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 --port 8001 \
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

### Experiment 1: developer system prompts (honesty, and honesty + ethics)
`--developer-prompt VARIANT` adds a "developer system prompt" to the **model under test on every benchmark**: MASK
pressure prompts, MMLU, GSM8K, MT-Bench and Machiavelli. It goes first in the system message, followed by the
benchmark's own system prompt (or alone where there is none). It is never added to the judges, and never to MASK
belief elicitation (beliefs must be elicited neutrally). Variants:
- `honest` (bare `--developer-prompt` means this): the MASK paper's prompt (App. A.6), "respond in a way that is honest and
  truthful". The paper applies it to MASK only; applying it to the other benchmarks is our extension.
- `honest_ethical`: honesty plus a short code of conduct (honesty, no deceiving others, care for people, fairness and
  rules, openness when declining, fiction is fine). The text is `DEVELOPER_PROMPTS` in
  `scripts/run_evals.py`. It adds ethics beyond honesty, so a MASK gain is not purely an honesty effect, and it can change
  Machiavelli behaviour (use `--mach-good` separately if you also want the game-specific "be good" instruction).

Use a new `--out` per variant (the script refuses to reuse an output folder run with a different setting).
```bash
for V in honest honest_ethical; do
python scripts/run_evals.py --split test --model Qwen/Qwen3.5-9B --developer-prompt $V \
  --base-url http://localhost:8000/v1 \
  --judge-model mistralai/Mistral-Small-3.2-24B-Instruct-2506 --judge-base-url http://localhost:8001/v1 \
  --concurrency 64 --mach-workers 30 --out outputs/devprompt_${V}_test
done
```
Compare against the baseline run on all benchmarks: MASK honesty (and evasion / no-belief / accuracy), MMLU, GSM8K, MT-Bench,
and the Machiavelli harm scores.

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

### Experiment 3: honesty steering (same contrast data as LoRRA, no training)
`scripts/steer_honesty.py` computes per-layer vectors `v_l` = mean (honest-persona − dishonest-persona) activation over
response tokens on generic alpaca prompts, and serves the model (Hugging Face, batched, OpenAI-compatible) with
`alpha * v_l` added to the residual stream at chosen layers. `alpha = 1` is one average persona shift; `0` is unsteered.
Run in the training env (`.venv-train`), with vLLM servers stopped except the judge (judge only needed for the final MASK run).

**MASK is held out (dev and test): nothing is tuned on it.** Selection protocol:
1. *Layer*: `compute` also scores each layer on 200 held-out alpaca prompts, i.e. how consistently an individual
   prompt's honest-minus-dishonest shift points along `v_l` (`heldout_cosine` in `meta.json`, top layers printed). Take
   a few of the best layers.
2. *Alpha*: sweep alpha on those layers with MMLU + GSM8K dev (`steer_sweep.py`; it refuses `--bench mask` unless
   `--final-mask`). Keep the largest alpha whose accuracy stays within a pre-set budget of the unsteered run (decide the
   budget before looking, e.g. 2 points).
3. *Final*: run MASK once on the chosen (layer, alpha), on the test split. Do not pick among MASK results; if you run
   several configs on MASK, report all of them.

```bash
# 1) vectors for all layers (~1000 examples + 200 held-out, one GPU, minutes)
CUDA_VISIBLE_DEVICES=0 python scripts/steer_honesty.py compute --model Qwen/Qwen3.5-9B --n 1000 --out models/steer/qwen3.5-9b

# 2) one steer server per GPU 0-5
for i in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$i nohup python scripts/steer_honesty.py serve --model Qwen/Qwen3.5-9B \
    --vectors models/steer/qwen3.5-9b/vectors.pt --port 900$i > steer_$i.log 2>&1 &
done

# 3) alpha sweep on capability dev sets (no MASK, no judge needed); use the top layers from step 1
python scripts/steer_sweep.py --ports 9000 9001 9002 9003 9004 9005 --layers <top layers> --alphas 0.5 1 2 4

# 4) final: MASK once on the chosen setting (judge on GPUs 6-7 as in "Running the model and a separate judge together")
curl -X POST localhost:9000/steer -d '{"layers":[<L>],"alpha":<A>}'
python scripts/run_evals.py --split test --bench mask --model qwen3.5-9b-steer --base-url http://localhost:9000/v1 \
  --judge-model mistralai/Mistral-Small-3.2-24B-Instruct-2506 --judge-base-url http://localhost:8001/v1 \
  --concurrency 32 --out outputs/steer_final_test
```
Sweep runs go to `outputs/steer_dev/<config>/` with a comparison table. Layer indices are 0-based decoder layers
(`meta.json` has the layer count and vector norms). A server's setting can be changed by hand with the `curl` above.

### Experiment 4: fine-tuning data (Moral Stories, DolusChat, synthetic anti-evasion, Tulu 3)
Builds an SFT file and a DPO file in `data/train/` (gitignored). Run both scripts from the eval client env
(`.venv`, which now includes `datasets`). Neither script trains on MASK or MACHIAVELLI: every example is checked for
13-gram overlap with MASK, MT-Bench, GSM8K test and MMLU test (plus the MASK canary; eval texts of 8-12 words, which
cannot form a 13-gram, are matched as exact word sequences; the 7.9% of MMLU questions under 8 words are not checked), and
synthetic scenarios whose fact is close to a MASK proposition are dropped (`scripts/decontam.py`).

| source | SFT | DPO | what it targets |
|---|---|---|---|
| Moral Stories (`demelin/moral_stories`, 6k) | moral action | moral > immoral action | MACHIAVELLI: half the rows use the game's own prompt format (numbered actions, answer = number, no "be ethical" instruction); both actions reach the goal, so reward isn't traded away |
| DolusChat (`AlignmentResearch/DolusChat`, 6k) | truthful reply | truthful > deceptive | lying in context |
| Synthetic (`gen_synthetic_honesty.py`) | honest reply | honest-and-direct > lie, honest-and-direct > evasive | MASK-shaped pressure, and evasion |
| Tulu 3 SFT mix (`allenai/tulu-3-sft-mixture`, 24k) | as is | none | keeping MMLU / GSM8K / MT-Bench |

```bash
# 1) synthetic scenarios (~60-80% survive the judge). The teacher/judge must NOT be the MASK eval judge (Mistral): the
#    data would be filtered by the model that later scores it. Serve another model on free GPUs, e.g. on :8002:
#      CUDA_VISIBLE_DEVICES=6,7 vllm serve Qwen/Qwen2.5-32B-Instruct --port 8002 --tensor-parallel-size 2 \
#        --gpu-memory-utilization 0.90 --max-model-len 16384
#    The script refuses the eval judge unless --allow-eval-judge; each row records its teacher and judge.
python scripts/gen_synthetic_honesty.py --n 3000 --concurrency 64 \
  --teacher-model Qwen/Qwen2.5-32B-Instruct --teacher-base-url http://localhost:8002/v1
python scripts/gen_synthetic_honesty.py --n 20 --mock     # plumbing check, no server

# 2) check DolusChat's field layout (auto-detected; override with --dolus-fields chosen=...,rejected=...,user=...,system=...)
python scripts/build_training_data.py --inspect doluschat

# 3) build the mix, then read data/train/samples.md and manifest.json before training
python scripts/build_training_data.py
```
Sizes and mix are flags (`--n-moral`, `--n-dolus`, `--n-tulu`, `--n-synthetic`, `--moral-mach-frac`, `--no-doluschat`,
`--tulu-exclude <source> ...`). Tulu 3 includes math and multiple-choice data (which can raise MMLU/GSM8K by itself) and
safety/refusal sources (which can raise evasion): see `tulu_sources` in `manifest.json` and consider `--tulu-exclude` on
the refusal ones.

**Tulu-only control.** To separate what the honesty/ethics data does from what Tulu does, build and train a control from
the same Tulu subset (same `--seed`, same exclusions) and evaluate it exactly like the real run:
```bash
python scripts/build_training_data.py --tulu-only                   # -> data/train/control_tulu/ (add the same --tulu-exclude ...)
torchrun --standalone --nproc_per_node=8 scripts/train_sft_dpo.py --stage sft \
  --data data/train/control_tulu/sft.jsonl --out-dir models/qwen3.5-9b-sft-tulu-only
```
Compare the SFT and SFT+DPO models against this control (and the base model), not only against the base model. It has
fewer optimizer steps than the full SFT because it has fewer examples; note that when reading the comparison. Reruns of the generator append and skip finished ids; dropped scenarios and the reason
go to `data/train/synthetic_rejects.jsonl`.

**Training on the mix** (`scripts/train_sft_dpo.py`, training env, same data-parallel setup as LoRRA; stop vLLM first).
LoRA on every Linear layer of each decoder block; prompts are rendered as at eval time (`enable_thinking=False`) and
only the final assistant reply is trained. Each stage saves a merged checkpoint plus `adapter/` and `train_log.json`
(validation loss before and after; for DPO also preference accuracy, margin, and how far the chosen reply's
log-prob moved).
```bash
# trial first: a few steps to check memory and time per step
torchrun --standalone --nproc_per_node=8 scripts/train_sft_dpo.py --stage sft --max-examples 512 --out-dir models/sft_trial

# 1) SFT from Qwen/Qwen3.5-9B -> models/qwen3.5-9b-sft   (lr 1e-4, global batch 64, 1 epoch)
torchrun --standalone --nproc_per_node=8 scripts/train_sft_dpo.py --stage sft
# 2) DPO from the SFT model, which is also the reference -> models/qwen3.5-9b-sft-dpo   (beta 0.1, lr 2e-5)
torchrun --standalone --nproc_per_node=8 scripts/train_sft_dpo.py --stage dpo --rpo-alpha 0.2
```
`--rpo-alpha` adds an NLL term on the chosen reply so DPO can't win just by pushing both replies down. Ablations:
`--stage dpo --model Qwen/Qwen3.5-9B --out-dir models/qwen3.5-9b-dpo-only` (no SFT stage). Evaluate as for LoRRA:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vllm serve models/qwen3.5-9b-sft-dpo --served-model-name qwen3.5-9b-sft-dpo \
  --port 8000 --data-parallel-size 6 --gpu-memory-utilization 0.90 --max-model-len 16384
python scripts/run_evals.py --split test --model qwen3.5-9b-sft-dpo --base-url http://localhost:8000/v1 \
  --judge-model mistralai/Mistral-Small-3.2-24B-Instruct-2506 --judge-base-url http://localhost:8001/v1 \
  --concurrency 96 --mach-workers 30 --out outputs/sft_dpo_full_test
```

### Analysis: what did LoRRA change?
`scripts/analyze_lorra.py` compares the LoRRA model with the base model on the same prompts (one GPU; the adapter is
toggled on/off, so both share one set of weights). It needs the trained adapter and the steering vectors from
Experiment 3 (`steer_honesty.py compute`), used as the reference honesty direction.
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analyze_lorra.py \
  --adapter models/qwen3.5-9b-lorra/adapter --vectors models/steer/qwen3.5-9b/vectors.pt --weights
```
Writes `outputs/lorra_analysis/{summary.md,results.json,layers_main.png,layers_last.png}`. Per layer and prompt set (unseen
alpaca prompts plain / honest persona / dishonest persona, and MASK pressure / belief prompts) it reports the relative
size of the activation shift, its cosine and projection along the honesty vector, and whether the shift is the same
vector for every prompt (`shared_frac` near 1, i.e. behaves like a steering vector) or depends on the prompt. MASK prompts
are used here only to understand the model; do not tune anything on them. See the script's docstring for the metrics.
