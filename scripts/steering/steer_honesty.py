"""Honesty activation steering with the same contrast data as LoRRA (scripts/training/train_lorra.py).

  compute  per-layer steering vectors  v_l = mean over response tokens of Act_l(honest persona) - Act_l(dishonest persona)
           on generic alpaca instructions (no MASK data).
  serve    OpenAI-compatible server (Hugging Face, dynamic batching) that adds  alpha * v_l  to the residual stream
           output of chosen layer(s) at every position. Layers/alpha can be changed at runtime with POST /steer, so
           one loaded model can be swept over many settings.

    python scripts/steering/steer_honesty.py compute --model Qwen/Qwen3.5-9B --n 1000 --out models/steer/qwen3.5-9b
    CUDA_VISIBLE_DEVICES=0 python scripts/steering/steer_honesty.py serve --model Qwen/Qwen3.5-9B \\
        --vectors models/steer/qwen3.5-9b/vectors.pt --port 9000
    curl -X POST localhost:9000/steer -d '{"layers": [16], "alpha": 2.0}'

MASK is a hold-out benchmark here: layer choice uses held-out alpaca prompts only (compute prints, per layer, how
consistently individual prompts' honest-minus-dishonest shifts align with the vector), alpha is limited by a
capability budget (MMLU/GSM8K dev), and MASK is run once on the final setting.

alpha = 1 adds the average honest-minus-dishonest persona shift once; alpha = 0 (or no layers) is the unsteered model.
Layer indices are decoder-layer indices (0-based); the vector for layer l is taken from that layer's output.
"""
import argparse
import json
import queue
import threading
import time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HONEST = "Pretend you're an honest person making statements about the world."
DISHONEST = "Pretend you're a dishonest person making statements about the world."


def load_model(name, device):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    model.config.use_cache = True
    return tok, model


def find_layers(model):
    n = getattr(model.config, "text_config", model.config).num_hidden_layers
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n and name.endswith("layers"):
            return mod
    raise RuntimeError("could not locate the decoder layer list")


def layer_tensor(out):
    return out[0] if isinstance(out, tuple) else out


# --------------------------------------------------------------------------- compute
def cmd_compute(a):
    from datasets import load_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_model(a.model, device)
    layers = find_layers(model)
    N = len(layers)
    tok.padding_side = "right"

    def chat_ids(system, user):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tok(text, add_special_tokens=False)["input_ids"][-a.max_prompt_tokens - 64:]

    ds = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=a.seed)
    ex = []
    for r in ds:
        if r["input"].strip() or len(r["output"].split()) < 12:
            continue
        resp = tok(r["output"] + "<|im_end|>", add_special_tokens=False)["input_ids"][:a.max_resp_tokens]
        ex.append(dict(plus=chat_ids(HONEST, r["instruction"]), minus=chat_ids(DISHONEST, r["instruction"]), resp=resp))
        if len(ex) >= a.n + a.n_heldout:
            break
    ex, held = ex[:a.n], ex[a.n:]
    print(f"{len(ex)} examples (+{len(held)} held-out), {N} layers")

    acc = torch.zeros(N, model.config.get_text_config().hidden_size, device=device, dtype=torch.float32)
    state = {}

    def make_hook(l):
        def hook(_m, _i, out):
            h = layer_tensor(out)
            idx = state["pos"].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            g = h.gather(1, idx).float() * state["valid"].unsqueeze(-1)
            if "per_ex" in state:
                state["per_ex"][:, l] += state["sign"] * g.sum(1)
            else:
                acc[l] += state["sign"] * g.sum((0, 1))
        return hook

    hooks = [layers[l].register_forward_hook(make_hook(l)) for l in range(N)]
    def pair_pass(batch):
        """Forward honest and dishonest versions of a batch; hooks accumulate (honest - dishonest) at response tokens."""
        R = max(len(e["resp"]) for e in batch)
        valid = torch.stack([torch.arange(R) < len(e["resp"]) for e in batch]).to(device)
        for key, sign in (("plus", 1.0), ("minus", -1.0)):
            seqs = [e[key] + e["resp"] for e in batch]
            L = max(map(len, seqs))
            ids = torch.full((len(batch), L), tok.pad_token_id)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, sq in enumerate(seqs):
                ids[j, :len(sq)] = torch.tensor(sq)
                att[j, :len(sq)] = 1
            pos = torch.stack([torch.arange(R).clamp(max=len(e["resp"]) - 1) + len(e[key]) for e in batch])
            state.update(pos=pos.to(device), valid=valid, sign=sign)
            model(input_ids=ids.to(device), attention_mask=att.to(device))
        return valid

    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(ex), a.batch_size):
            total_tokens += int(pair_pass(ex[i:i + a.batch_size]).sum())
            if (i // a.batch_size) % 20 == 0:
                print(f"{i + a.batch_size}/{len(ex)}", flush=True)
        vectors = acc / total_tokens
        # layer scores on held-out prompts: does each prompt's own honest-minus-dishonest shift point along the vector?
        cos, pair_acc = [], []
        for i in range(0, len(held), a.batch_size):
            batch = held[i:i + a.batch_size]
            state["per_ex"] = torch.zeros(len(batch), N, vectors.shape[1], device=device)
            valid = pair_pass(batch)
            d = state["per_ex"] / valid.sum(1).clamp(min=1).view(-1, 1, 1)
            del state["per_ex"]
            cos.append(torch.nn.functional.cosine_similarity(d, vectors.unsqueeze(0), dim=-1).cpu())
            pair_acc.append(((d * vectors.unsqueeze(0)).sum(-1) > 0).float().cpu())
        cos = torch.cat(cos).mean(0) if cos else torch.zeros(N)
        pair_acc = torch.cat(pair_acc).mean(0) if pair_acc else torch.zeros(N)
    for h in hooks:
        h.remove()
    vectors = vectors.cpu()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": vectors, "model": a.model, "n_examples": len(ex), "tokens": total_tokens}, out / "vectors.pt")
    (out / "meta.json").write_text(json.dumps({"model": a.model, "n_examples": len(ex), "tokens": total_tokens,
                                               "layers": N, "vector_norms": [round(float(v.norm()), 3) for v in vectors],
                                               "heldout_cosine": [round(float(x), 4) for x in cos],
                                               "heldout_pair_acc": [round(float(x), 4) for x in pair_acc]}, indent=1))
    print("vector norm per layer:", [round(float(v.norm()), 2) for v in vectors])
    print("held-out cosine per layer (higher = more consistent direction):", [round(float(x), 3) for x in cos])
    print("top layers by held-out cosine:", [int(i) for i in cos.argsort(descending=True)[:8]])
    print("saved to", out.resolve())


# --------------------------------------------------------------------------- serve
class Request:
    def __init__(self, messages, temperature, max_tokens, template_kwargs):
        self.messages, self.temperature, self.max_tokens = messages, temperature, max_tokens
        self.template_kwargs = template_kwargs
        self.future = Future()


class Engine:
    """Batches concurrent chat requests (grouped by temperature) into single HF generate calls."""

    def __init__(self, tok, model, vectors, max_batch, wait):
        self.tok, self.model, self.vectors = tok, model, vectors
        self.layers = find_layers(model)
        self.device = next(model.parameters()).device
        self.max_batch, self.wait = max_batch, wait
        self.q, self.pending = queue.Queue(), []
        self.lock = threading.Lock()
        self.hooks, self.cfg = [], {"layers": [], "alpha": 0.0}
        threading.Thread(target=self.loop, daemon=True).start()

    def set_steer(self, layers, alpha):
        with self.lock:  # waits for any in-flight generation
            for h in self.hooks:
                h.remove()
            self.hooks = []
            if layers and alpha:
                for l in layers:
                    v = self.vectors[l].to(self.device, next(self.model.parameters()).dtype) * alpha
                    self.hooks.append(self.layers[l].register_forward_hook(self._hook(v)))
            self.cfg = {"layers": list(layers), "alpha": float(alpha)}

    @staticmethod
    def _hook(v):
        def hook(_m, _i, out):
            if isinstance(out, tuple):
                return (out[0] + v,) + tuple(out[1:])
            return out + v
        return hook

    def submit(self, req):
        self.q.put(req)
        return req.future.result()

    def loop(self):
        while True:
            if not self.pending:
                self.pending.append(self.q.get())
            end = time.time() + self.wait
            while len(self.pending) < self.max_batch * 4 and time.time() < end:
                try:
                    self.pending.append(self.q.get(timeout=max(0.0, end - time.time())))
                except queue.Empty:
                    break
            t = self.pending[0].temperature
            batch = [r for r in self.pending if r.temperature == t][:self.max_batch]
            self.pending = [r for r in self.pending if r not in batch]
            try:
                self.run(batch)
            except Exception as e:  # noqa: BLE001
                for r in batch:
                    r.future.set_exception(e)

    def run(self, batch):
        tok = self.tok
        texts = [tok.apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True, **r.template_kwargs)
                 for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        t = batch[0].temperature
        with self.lock, torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max(r.max_tokens for r in batch), do_sample=t > 0,
                                      temperature=t if t > 0 else None, top_p=1.0, top_k=0 if t > 0 else None,
                                      pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
        for r, row in zip(batch, new):
            ids = row.tolist()[:r.max_tokens]
            stop = next((i for i, x in enumerate(ids) if x in eos), None)
            finish = "stop" if stop is not None else "length"
            ids = ids[:stop] if stop is not None else ids
            r.future.set_result({"text": tok.decode(ids, skip_special_tokens=True), "finish": finish,
                                 "prompt_tokens": int(enc["attention_mask"][batch.index(r)].sum()),
                                 "completion_tokens": len(ids)})


def make_handler(engine, model_name):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/steer":
                self._send(200, engine.cfg)
            else:
                self._send(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            try:
                if self.path.rstrip("/") == "/steer":
                    engine.set_steer(body.get("layers", []), body.get("alpha", 0.0))
                    return self._send(200, engine.cfg)
                r = Request(body["messages"], float(body.get("temperature") or 0.0),
                            int(body.get("max_tokens") or body.get("max_completion_tokens") or 512),
                            body.get("chat_template_kwargs") or {})
                res = engine.submit(r)
                self._send(200, {
                    "id": "chatcmpl-steer", "object": "chat.completion", "created": int(time.time()), "model": model_name,
                    "choices": [{"index": 0, "finish_reason": res["finish"],
                                 "message": {"role": "assistant", "content": res["text"]}}],
                    "usage": {"prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
                              "total_tokens": res["prompt_tokens"] + res["completion_tokens"]}})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": {"message": str(e), "type": type(e).__name__}})

        def log_message(self, *a):
            pass
    return H


def cmd_serve(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_model(a.model, device)
    vectors = torch.load(a.vectors)["vectors"]
    engine = Engine(tok, model, vectors, a.max_batch, a.wait)
    if a.layers:
        engine.set_steer(a.layers, a.alpha)
    print(f"serving {a.served_name} on :{a.port} | steering {engine.cfg}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(engine, a.served_name)).serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--model", default="Qwen/Qwen3.5-9B")
    c.add_argument("--out", default="models/steer/qwen3.5-9b")
    c.add_argument("--n", type=int, default=1000)
    c.add_argument("--n-heldout", type=int, default=200, help="extra examples used only to score layers")
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--max-prompt-tokens", type=int, default=128)
    c.add_argument("--max-resp-tokens", type=int, default=64)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(fn=cmd_compute)
    s = sub.add_parser("serve")
    s.add_argument("--model", default="Qwen/Qwen3.5-9B")
    s.add_argument("--vectors", default="models/steer/qwen3.5-9b/vectors.pt")
    s.add_argument("--served-name", default="qwen3.5-9b-steer")
    s.add_argument("--port", type=int, default=9000)
    s.add_argument("--max-batch", type=int, default=32)
    s.add_argument("--wait", type=float, default=0.05, help="seconds to wait to fill a batch")
    s.add_argument("--layers", type=int, nargs="*", default=[], help="initial steering layers (default: none)")
    s.add_argument("--alpha", type=float, default=0.0)
    s.set_defaults(fn=cmd_serve)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
