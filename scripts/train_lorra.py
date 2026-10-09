"""LoRRA honesty training (MASK paper §5.2; Zou et al. 2023), data-parallel over all GPUs.

Script version of notebooks/lorra_honesty.ipynb (same objective and config). Each rank holds a full
bf16 copy of the model and trains on its own slice of every global batch; only the (small) LoRA gradients
are all-reduced. Frozen-model activations (the targets) are computed on the fly with the adapter disabled.

    torchrun --nproc_per_node=8 scripts/train_lorra.py                       # all GPUs
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_lorra.py   # a subset
    python scripts/train_lorra.py --n-train 256 --epochs 1                   # single GPU

Output: <out-dir>/ (merged, vLLM-loadable) and <out-dir>/adapter/, plus loss_log.json and lorra_config.json.
The paper gives no rank/layers/alpha/data/steps, so those are config choices (RepE-style defaults).
"""
import argparse
import json
import math
import os
import random
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HONEST = "Pretend you're an honest person making statements about the world."
DISHONEST = "Pretend you're a dishonest person making statements about the world."


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--out-dir", default="models/qwen3.5-9b-lorra")
    # data
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--max-prompt-tokens", type=int, default=128)
    p.add_argument("--max-resp-tokens", type=int, default=64)
    # LoRRA
    p.add_argument("--alpha", type=float, default=5.0, help="strength of the honesty contrast vector")
    p.add_argument("--target-fracs", type=float, nargs="+", default=[0.5, 0.75], help="target layers L_t, as depth fractions")
    p.add_argument("--edit-range", type=float, nargs=2, default=[0.15, 0.5],
                   help="editable layers L_e as [lo, hi) depth fractions; must sit before the first target layer")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # optimisation
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--global-batch", type=int, default=16, help="sequences per optimizer step across all ranks")
    p.add_argument("--micro-batch", type=int, default=2, help="sequences per rank per forward/backward")
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-heldout", type=int, default=64, help="examples excluded from training for the sanity check")
    return p.parse_args()


def main():
    a = parse_args()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    dev = torch.device(f"cuda:{local}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(local)
    if world > 1:
        # long timeout: rank 0 alone runs the sanity check and saves the merged model while the others wait
        dist.init_process_group("nccl" if use_cuda else "gloo", timeout=timedelta(minutes=60),
                                device_id=dev if use_cuda else None)
    is_main = rank == 0

    def log(*m):
        if is_main:
            print(*m, flush=True)

    per_step = a.micro_batch * world
    if a.global_batch % per_step:
        raise SystemExit(f"--global-batch {a.global_batch} must be a multiple of micro-batch x world size = {per_step}")
    accum = a.global_batch // per_step

    random.seed(a.seed)
    torch.manual_seed(a.seed)  # same on every rank -> identical LoRA init
    tok = AutoTokenizer.from_pretrained(a.model)

    def chat_ids(system, user):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tok(text, add_special_tokens=False)["input_ids"][-a.max_prompt_tokens - 64:]

    # ---- data (rank 0 downloads first to avoid a cache race)
    if world > 1 and not is_main:
        dist.barrier()
    ds = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=a.seed)
    if world > 1 and is_main:
        dist.barrier()
    examples = []
    for r in ds:
        if r["input"].strip() or len(r["output"].split()) < 12:
            continue
        resp = tok(r["output"] + "<|im_end|>", add_special_tokens=False)["input_ids"][:a.max_resp_tokens]
        examples.append(dict(neutral=chat_ids(None, r["instruction"]), plus=chat_ids(HONEST, r["instruction"]),
                             minus=chat_ids(DISHONEST, r["instruction"]), resp=resp))
        if len(examples) >= a.n_train + a.n_heldout:
            break
    held, train = examples[-a.n_heldout:], examples[:-a.n_heldout]
    steps_per_epoch = len(train) // a.global_batch
    total = steps_per_epoch * a.epochs
    log(f"{len(train)} train + {len(held)} held-out examples | world={world} micro={a.micro_batch} accum={accum} "
        f"global batch={a.global_batch} | {total} optimizer steps")

    # ---- model
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if use_cuda else torch.float32,
                                                 device_map={"": dev.index if use_cuda else "cpu"})
    model.config.use_cache = False
    cfg = getattr(model.config, "text_config", model.config)
    N = cfg.num_hidden_layers
    TARGET = sorted({min(N - 1, round(f * N)) for f in a.target_fracs})
    EDIT = list(range(int(a.edit_range[0] * N), int(a.edit_range[1] * N)))
    assert max(EDIT) <= min(TARGET), "editable layers must come before the target layers"
    log(f"{N} layers | target layers {TARGET} | editable layers {EDIT[0]}..{EDIT[-1]}")

    # PEFT forbids layers_to_transform with a string target_modules, so list the exact Linear modules of the
    # editable decoder layers (works for Qwen3.5's mixed full-attention / linear-attention layers too)
    prefix = next(n for n, m in model.named_modules()
                  if isinstance(m, torch.nn.ModuleList) and len(m) == N and n.endswith("layers"))
    layer_list = model.get_submodule(prefix)
    targets = [f"{prefix}.{i}.{n}" for i in EDIT for n, m in layer_list[i].named_modules()
               if isinstance(m, torch.nn.Linear)]
    log(f"LoRA on {len(targets)} Linear modules in layers {EDIT[0]}..{EDIT[-1]} (e.g. {targets[0]}, {targets[-1]})")
    pm = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                                          target_modules=targets, task_type="CAUSAL_LM"))
    if is_main:
        pm.print_trainable_parameters()
    params = [p for p in pm.parameters() if p.requires_grad]
    if world > 1:  # make sure every rank starts from rank 0's adapter
        for p in params:
            dist.broadcast(p.data, src=0)

    def collate(batch):
        R = max(len(e["resp"]) for e in batch)
        out = {}
        for k in ("neutral", "plus", "minus"):
            seqs = [e[k] + e["resp"] for e in batch]
            L = max(map(len, seqs))
            ids = torch.full((len(batch), L), tok.pad_token_id)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, :len(s)] = torch.tensor(s)
                att[i, :len(s)] = 1
            pos = torch.stack([torch.arange(R).clamp(max=len(e["resp"]) - 1) + len(e[k]) for e in batch])
            out[k] = (ids, att, pos)
        out["valid"] = torch.stack([torch.arange(R) < len(e["resp"]) for e in batch])
        return out

    def resp_acts(m, ids, att, pos):
        """Hidden states at the target layers over response positions -> list of [B, R, H]."""
        hs = m(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True).hidden_states
        idx = pos.to(dev).unsqueeze(-1).expand(-1, -1, hs[0].shape[-1])
        return [hs[l + 1].gather(1, idx) for l in TARGET]   # hs[0] is the embedding output

    def lorra_loss(b):
        with pm.disable_adapter(), torch.no_grad():
            a0 = resp_acts(pm, *b["neutral"])
            ap = resp_acts(pm, *b["plus"])
            am = resp_acts(pm, *b["minus"])
            targets = [x.float() + a.alpha * (p.float() - n.float()) for x, p, n in zip(a0, ap, am)]
        cur = resp_acts(pm, *b["neutral"])
        v = b["valid"].to(dev).unsqueeze(-1)
        loss = 0
        for c, t in zip(cur, targets):
            loss = loss + (((c.float() - t) ** 2) * v).sum() / (v.sum() * c.shape[-1])
        return loss / len(TARGET)

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / max(1, a.warmup_frac * total)) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))

    # ---- training
    pm.train()
    log_steps, step = [], 0
    for ep in range(a.epochs):
        order = list(range(len(train)))
        random.Random(a.seed + ep).shuffle(order)  # identical order on every rank
        for s in range(steps_per_epoch):
            base = s * a.global_batch
            step_loss = torch.zeros((), device=dev)
            for g in range(accum):
                off = base + g * per_step + rank * a.micro_batch
                batch = [train[order[i]] for i in range(off, off + a.micro_batch)]
                loss = lorra_loss(collate(batch)) / accum
                loss.backward()
                step_loss += loss.detach()
            if world > 1:  # average LoRA grads (and the logged loss) across ranks
                flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
                dist.all_reduce(flat)
                flat /= world
                i = 0
                for p in params:
                    p.grad = flat[i:i + p.numel()].view_as(p).to(p.dtype).clone()
                    i += p.numel()
                dist.all_reduce(step_loss)
                step_loss /= world
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            log_steps.append(step_loss.item())
            if step % 10 == 0 or step == total:
                log(f"step {step}/{total} loss {sum(log_steps[-10:]) / len(log_steps[-10:]):.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e}")

    # ---- sanity check on held-out examples (rank 0): adapter shift should align with the honesty direction
    if is_main:
        pm.eval()
        cos = []
        with torch.no_grad():
            for i in range(0, len(held), 8):
                b = collate(held[i:i + 8])
                v = b["valid"].to(dev)
                with pm.disable_adapter():
                    a0 = resp_acts(pm, *b["neutral"])
                    ap = resp_acts(pm, *b["plus"])
                    am = resp_acts(pm, *b["minus"])
                cur = resp_acts(pm, *b["neutral"])
                for c, x, p, n in zip(cur, a0, ap, am):
                    c_ = torch.nn.functional.cosine_similarity((c - x).float(), (p - n).float(), dim=-1)
                    cos.append(c_[v].mean().item())
        print("mean cosine(adapter shift, honesty contrast vector):", sum(cos) / max(1, len(cos)), flush=True)
        msgs = [{"role": "user", "content": "Is it true that humans only use 10% of their brains?"}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                      return_tensors="pt", return_dict=True).to(dev)
        print(tok.decode(pm.generate(**ids, max_new_tokens=120, do_sample=False)[0][ids["input_ids"].shape[1]:],
                         skip_special_tokens=True), flush=True)

        # ---- save adapter + merged checkpoint (rank 0 only)
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pm.save_pretrained(out / "adapter")
        merged = pm.merge_and_unload()
        merged.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)
        (out / "lorra_config.json").write_text(json.dumps(
            {**vars(a), "world_size": world, "target_layers": TARGET, "edit_layers": EDIT}, indent=2))
        (out / "loss_log.json").write_text(json.dumps(log_steps))
        print("saved to", out.resolve(), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
