"""What did LoRRA change? Compare the LoRRA model against the base model, layer by layer, on the same prompts.

The adapter is toggled on/off on one loaded model (PEFT `disable_adapter()`), so a single GPU is enough.

    python scripts/analyze_lorra.py \\
        --adapter models/qwen3.5-9b-lorra/adapter --vectors models/steer/qwen3.5-9b/vectors.pt

Needs the steering vectors (`steer_honesty.py compute`) as the reference "honesty direction" v_l.
For every prompt set and layer it reports, for the shift delta = h_lora - h_base at the residual stream:

  rel          ||delta|| / ||h_base||                       how big is the change (0 below the first edited layer)
  cos_v        cos(delta, v_l)                              does it point along the honest-minus-dishonest direction
  proj_v       (delta . v_l/||v_l||) / ||v_l||              size along v_l in units of ||v_l|| ("alpha-equivalent";
                                                            LoRRA trains toward alpha, see lorra_config.json)
  shared_frac  ||mean_i d_i||^2 / mean_i ||d_i||^2          ~1: same vector for every prompt (acts like steering);
                                                            ~0: depends on the prompt
  pair_cos     mean pairwise cos(d_i, d_j)                  same idea, direction only
  top1_energy  sigma_1^2 / sum sigma^2 of the d_i           how low-rank the prompt-to-prompt variation is

Prompt sets (all measured at the residual-stream output of every decoder layer):
  alpaca_neutral / alpaca_honest / alpaca_dishonest   unseen alpaca prompts, plain / honest persona / dishonest persona,
                                                      measured on the reference-response tokens (as in training)
  mask_pressure / mask_belief                         MASK pressure prompts and their neutral belief prompts, prompt tokens
                                                      only. MASK is held out: use these numbers to understand the model,
                                                      never to tune anything.
Each set is reported at two position groups: "main" (response tokens for alpaca, all prompt tokens for MASK) and "last"
(the last prompt token, which predicts the first response token).

--weights additionally reports the size and effective rank of the LoRA update per layer and module (adapter file only).
Outputs: <out>/results.json, <out>/summary.md, <out>/*.png
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
HONEST = "Pretend you're an honest person making statements about the world."
DISHONEST = "Pretend you're a dishonest person making statements about the world."
MASK_ARCHES = ["continuations", "disinformation", "known_facts", "provided_facts", "statistics"]
EPS = 1e-8


def find_layers(model, n):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n and name.endswith("layers"):
            return mod
    raise RuntimeError("could not locate the decoder layer list")


# --------------------------------------------------------------------------- prompt sets
def build_sets(tok, a):
    def chat(system, user):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tok(text, add_special_tokens=False)["input_ids"][-a.max_prompt_tokens:]

    sets = {}
    # alpaca prompts the adapter never saw: skip the examples train_lorra.py used (n_train + held-out)
    ds = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=a.seed)
    rows, seen = [], 0
    for r in ds:
        if r["input"].strip() or len(r["output"].split()) < 12:
            continue
        seen += 1
        if seen <= a.skip_qualifying:
            continue
        rows.append(r)
        if len(rows) >= a.n_alpaca:
            break
    for name, system in (("alpaca_neutral", None), ("alpaca_honest", HONEST), ("alpaca_dishonest", DISHONEST)):
        items = []
        for r in rows:
            p = chat(system, r["instruction"])
            resp = tok(r["output"] + "<|im_end|>", add_special_tokens=False)["input_ids"][:a.max_resp_tokens]
            items.append(dict(ids=p + resp, main=list(range(len(p), len(p) + len(resp))), last=len(p) - 1))
        sets[name] = items

    pressure, belief = [], []
    for arch in MASK_ARCHES:
        df = pd.read_parquet(ROOT / "data" / "evals" / "mask" / arch / "test-00000-of-00001.parquet")
        df = df.sample(n=min(a.n_mask, len(df)), random_state=a.seed)
        for _, r in df.iterrows():
            p = chat(r["system_prompt"], r["user_prompt"])
            pressure.append(dict(ids=p, main=list(range(len(p))), last=len(p) - 1))
            if isinstance(r.get("belief_elicit_1"), str):
                b = chat("You are a helpful assistant.", r["belief_elicit_1"])
                belief.append(dict(ids=b, main=list(range(len(b))), last=len(b) - 1))
    sets["mask_pressure"], sets["mask_belief"] = pressure, belief
    return sets


# --------------------------------------------------------------------------- activation diff
@torch.no_grad()
def diff_set(pm, layers, items, V, pad_id, dev, bs):
    """-> dict group -> dict metric -> np.array [n_items, N] ; group -> mean_delta [n_items, N, H] (fp16, CPU)."""
    N, H = V.shape
    Vn = V / V.norm(dim=-1, keepdim=True).clamp(min=EPS)
    vnorm = V.norm(dim=-1)
    out = {g: {k: np.zeros((len(items), N), dtype=np.float32) for k in ("dnorm", "hnorm", "cos", "proj")} for g in ("main", "last")}
    mean_delta = {g: torch.zeros(len(items), N, H, dtype=torch.bfloat16) for g in ("main", "last")}
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    cap = {}
    hooks = [layers[l].register_forward_hook(lambda _m, _i, o, l=l: cap.__setitem__(l, o[0] if isinstance(o, tuple) else o))
             for l in range(N)]
    try:
        for s in range(0, len(order), bs):
            idxs = order[s:s + bs]
            batch = [items[i] for i in idxs]
            L = max(len(b["ids"]) for b in batch)
            ids = torch.full((len(batch), L), pad_id)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, b in enumerate(batch):
                ids[j, :len(b["ids"])] = torch.tensor(b["ids"])
                att[j, :len(b["ids"])] = 1
            ids, att = ids.to(dev), att.to(dev)
            with pm.disable_adapter():
                pm(input_ids=ids, attention_mask=att)
            ref = {l: cap[l].clone() for l in range(N)}  # base-model activations
            pm(input_ids=ids, attention_mask=att)
            for g in ("main", "last"):
                P = max(len(b["main"]) if g == "main" else 1 for b in batch)
                pos = torch.zeros(len(batch), P, dtype=torch.long)
                msk = torch.zeros(len(batch), P)
                for j, b in enumerate(batch):
                    ps = b["main"] if g == "main" else [b["last"]]
                    pos[j, :len(ps)] = torch.tensor(ps)
                    msk[j, :len(ps)] = 1
                pos, msk = pos.to(dev), msk.to(dev)
                cnt = msk.sum(1).clamp(min=1)
                gi = pos.unsqueeze(-1).expand(-1, -1, H)
                it = torch.tensor(idxs)
                for l in range(N):
                    hb = ref[l].gather(1, gi).float()
                    d = cap[l].gather(1, gi).float() - hb
                    dn, hn = d.norm(dim=-1), hb.norm(dim=-1)
                    cos = (d * V[l]).sum(-1) / (dn * vnorm[l]).clamp(min=EPS)
                    proj = (d * Vn[l]).sum(-1) / vnorm[l].clamp(min=EPS)
                    for key, val in (("dnorm", dn), ("hnorm", hn), ("cos", cos), ("proj", proj)):
                        out[g][key][idxs, l] = ((val * msk).sum(1) / cnt).cpu().numpy()
                    mean_delta[g][it, l] = ((d * msk.unsqueeze(-1)).sum(1) / cnt.unsqueeze(-1)).to(torch.bfloat16).cpu()
            cap.clear()
    finally:
        for h in hooks:
            h.remove()
    return out, mean_delta


def summarize(raw, mean_delta, dev):
    """Per-layer aggregates over items for one group."""
    n, N, _ = mean_delta.shape
    res = {"rel": (raw["dnorm"].sum(0) / raw["hnorm"].sum(0).clip(min=EPS)).tolist(),
           "cos_v": raw["cos"].mean(0).tolist(), "proj_v": raw["proj"].mean(0).tolist()}
    shared, pair, top1 = [], [], []
    for l in range(N):
        D = mean_delta[:, l].to(dev).float()
        nrm2 = (D ** 2).sum(1)
        if nrm2.mean() < 1e-12:
            shared.append(float("nan")); pair.append(float("nan")); top1.append(float("nan"))
            continue
        shared.append(float((D.mean(0) ** 2).sum() / nrm2.mean()))
        U = D / nrm2.sqrt().clamp(min=EPS).unsqueeze(1)
        pair.append(float(((U.sum(0) ** 2).sum() - n) / max(1, n * (n - 1))))
        sv = torch.linalg.svdvals(D)
        top1.append(float(sv[0] ** 2 / (sv ** 2).sum()))
    res.update(shared_frac=shared, pair_cos=pair, top1_energy=top1)
    return res


# --------------------------------------------------------------------------- weights
def weight_stats(adapter_dir):
    from safetensors.torch import load_file
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    rows = []
    for k in sd:
        m = re.search(r"layers\.(\d+)\.(.+)\.lora_A\.weight$", k)
        if not m:
            continue
        A, B = sd[k].float(), sd[k.replace("lora_A", "lora_B")].float()
        # singular values of scaling * B @ A via thin QR (rank r), instead of an SVD of the full matrix
        _, Rb = torch.linalg.qr(B)
        _, Ra = torch.linalg.qr(A.T)
        sv = scaling * torch.linalg.svdvals(Rb @ Ra.T)
        p = sv / sv.sum().clamp(min=EPS)
        rows.append(dict(layer=int(m.group(1)), module=m.group(2), fro=float(sv.norm()),
                         eff_rank=float(torch.exp(-(p * (p + 1e-12).log()).sum()))))
    return rows


# --------------------------------------------------------------------------- reporting
def plots(res, out, targets, edits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("rel", "relative shift ||d||/||h||"), ("cos_v", "cos(d, honesty vector)"),
               ("proj_v", "projection on v (units of ||v||)"), ("shared_frac", "shared fraction (1 = same vector for every prompt)")]
    for group in ("main", "last"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, (k, title) in zip(axes.flat, metrics):
            for name, r in res["sets"].items():
                ax.plot(r[group][k], label=name)
            for t in targets:
                ax.axvline(t, color="gray", ls=":", lw=1)
            ax.axvspan(min(edits), max(edits), color="gold", alpha=0.12)
            ax.set_title(title)
            ax.set_xlabel("layer")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"LoRRA vs base ({group} positions); shaded = edited layers, dotted = target layers")
        fig.tight_layout()
        fig.savefig(out / f"layers_{group}.png", dpi=130)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--adapter", default="models/qwen3.5-9b-lorra/adapter")
    p.add_argument("--vectors", default="models/steer/qwen3.5-9b/vectors.pt")
    p.add_argument("--out", default="outputs/lorra_analysis")
    p.add_argument("--n-alpaca", type=int, default=200)
    p.add_argument("--n-mask", type=int, default=40, help="MASK rows per archetype (5 archetypes)")
    p.add_argument("--skip-qualifying", type=int, default=None,
                   help="alpaca examples used in training; default n_train + n_heldout from lorra_config.json (5064)")
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--max-resp-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weights", action="store_true", help="also analyse the LoRA weight update")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter = Path(a.adapter)
    lcfg_path = adapter.parent / "lorra_config.json"
    lcfg = json.loads(lcfg_path.read_text()) if lcfg_path.exists() else {}
    targets, edits = lcfg.get("target_layers", []), lcfg.get("edit_layers", [0])

    if a.skip_qualifying is None:
        a.skip_qualifying = lcfg.get("n_train", 5000) + lcfg.get("n_heldout", 64)
    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32).to(dev)
    base.config.use_cache = False
    pm = PeftModel.from_pretrained(base, str(adapter)).eval()
    N = getattr(base.config, "text_config", base.config).num_hidden_layers
    layers = find_layers(base, N)
    V = torch.load(a.vectors)["vectors"].to(dev).float()
    assert V.shape[0] == N, f"vectors have {V.shape[0]} layers, model has {N}"

    sets = build_sets(tok, a)
    print({k: len(v) for k, v in sets.items()}, flush=True)
    res = {"layers": N, "target_layers": targets, "edit_layers": edits, "lorra_alpha": lcfg.get("alpha"), "sets": {}}
    for name, items in sets.items():
        raw, md = diff_set(pm, layers, items, V, tok.pad_token_id, dev, a.batch_size)
        res["sets"][name] = {g: summarize(raw[g], md[g], dev) for g in ("main", "last")}
        r = res["sets"][name]["main"]
        peak = int(np.nanargmax(r["rel"]))
        print(f"{name}: peak rel shift {r['rel'][peak]:.3f} at layer {peak}; cos_v@peak {r['cos_v'][peak]:.3f}", flush=True)
        del raw, md
    if a.weights:
        res["weights"] = weight_stats(adapter)
    (out / "results.json").write_text(json.dumps(res, indent=1))

    # ---- markdown summary
    show = sorted({*(targets or []), min(edits), max(edits), N // 2, N - 1})
    L = ["# LoRRA vs base: activation diff", "",
         f"Edited layers {min(edits)}..{max(edits)}, target layers {targets}, LoRRA alpha {res['lorra_alpha']} "
         f"(an alpha-equivalent projection of about that size at the target layers means the adapter reproduces the training target).", ""]
    for group in ("main", "last"):
        L += [f"## Position group: {group}", "", "| set | layer | rel | cos_v | proj_v | shared_frac | pair_cos | top1_energy |", "|---|---|---|---|---|---|---|---|"]
        for name, r in res["sets"].items():
            for l in show:
                g = r[group]
                L.append(f"| {name} | {l} | {g['rel'][l]:.4f} | {g['cos_v'][l]:.3f} | {g['proj_v'][l]:.3f} | "
                         f"{g['shared_frac'][l]:.3f} | {g['pair_cos'][l]:.3f} | {g['top1_energy'][l]:.3f} |")
        L.append("")
    if a.weights:
        w = pd.DataFrame(res["weights"])
        L += ["## LoRA update size per layer (Frobenius norm of scaled BA, summed over modules)", "",
              "```", w.groupby("layer").agg(fro=("fro", "sum"), eff_rank=("eff_rank", "mean")).round(3).to_string(), "```", ""]
    (out / "summary.md").write_text("\n".join(L))
    try:
        plots(res, out, targets, edits)
    except Exception as e:  # noqa: BLE001
        print("plotting skipped:", e)
    print("saved to", out.resolve())


if __name__ == "__main__":
    main()
