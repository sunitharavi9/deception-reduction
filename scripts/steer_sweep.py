"""Sweep steering settings on capability dev sets: one steer server per GPU, one run_evals.py run per (layers, alpha).
MASK is a hold-out benchmark, so it is NOT run by default (--bench defaults to mmlu gsm8k); use it only on the final
setting, with --bench mask --final-mask.

    python scripts/steer_sweep.py --ports 9000 9001 9002 9003 9004 9005 --layers 12 16 20 --alphas 0.5 1 2

Each server must already be running (`steer_honesty.py serve --port <p>`). The sweep switches a server's setting with
POST /steer, runs `run_evals.py --split dev` into <out-root>/L<layers>_a<alpha>/, then prints a comparison table.
The unsteered run (alpha 0) is always included as the reference.
"""
import argparse
import itertools
import json
import queue
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def post(url, obj):
    req = urllib.request.Request(url, json.dumps(obj).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def tag(layers, alpha):
    return "unsteered" if not layers or not alpha else f"L{'-'.join(map(str, layers))}_a{alpha:g}"


def table(out_root, configs):
    rows = []
    for layers, alpha in configs:
        d = out_root / tag(layers, alpha)
        row = {"config": tag(layers, alpha)}
        try:
            m = json.loads((d / "mask_metrics.json").read_text())
            row["honesty"], row["honesty_norm"] = m["macro_honesty"], m["macro_honesty_normalized"]
            row["no_belief"] = m["overall"]["no_belief_rate"]
        except Exception:  # noqa: BLE001
            pass
        try:
            c = json.loads((d / "capability_metrics.json").read_text())
            for k in ("mmlu", "gsm8k"):
                if k in c:
                    row[k] = c[k]["accuracy"]
            if "mt_bench" in c:
                row["mt_bench"] = c["mt_bench"]["score"]
        except Exception:  # noqa: BLE001
            pass
        rows.append(row)
    cols = ["config", "honesty", "honesty_norm", "no_belief", "mmlu", "gsm8k", "mt_bench"]
    cols = [c for c in cols if any(c in r for r in rows)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(r.get(c, "") if isinstance(r.get(c, ""), str) else
                                       (f"{r[c]:.3f}" if c in r and r[c] is not None else "n/a") for c in cols) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ports", type=int, nargs="+", required=True, help="steer servers (one config at a time each)")
    p.add_argument("--host", default="localhost")
    p.add_argument("--layers", type=int, nargs="+", required=True, help="each layer is swept on its own")
    p.add_argument("--alphas", type=float, nargs="+", required=True)
    p.add_argument("--model-name", default="qwen3.5-9b-steer", help="must match the server's --served-name")
    p.add_argument("--bench", nargs="+", default=["mmlu", "gsm8k"])
    p.add_argument("--final-mask", action="store_true",
                   help="allow --bench mask (MASK is held out; use for the single final configuration only)")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--judge-model", default=None, help="needed for mask / mt_bench")
    p.add_argument("--judge-base-url", default=None)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--out-root", default="outputs/steer_dev")
    a = p.parse_args()

    if "mask" in a.bench and not a.final_mask:
        p.error("MASK is held out: select layer/alpha without it, then run the final setting with --bench mask --final-mask")
    if ("mask" in a.bench or "mt_bench" in a.bench) and not (a.judge_model and a.judge_base_url):
        p.error("--judge-model and --judge-base-url are required for mask / mt_bench")
    out_root = ROOT / a.out_root if not Path(a.out_root).is_absolute() else Path(a.out_root)
    configs = [([], 0.0)] + [([l], al) for l, al in itertools.product(a.layers, a.alphas)]
    work = queue.Queue()
    for c in configs:
        work.put(c)

    def worker(port):
        base = f"http://{a.host}:{port}"
        while True:
            try:
                layers, alpha = work.get_nowait()
            except queue.Empty:
                return
            out = out_root / tag(layers, alpha)
            post(f"{base}/steer", {"layers": layers, "alpha": alpha})
            print(f"[:{port}] {tag(layers, alpha)} -> {out}", flush=True)
            cmd = [sys.executable, str(ROOT / "scripts" / "run_evals.py"), "--split", a.split, "--bench", *a.bench,
                   "--model", a.model_name, "--base-url", f"{base}/v1",
                   "--concurrency", str(a.concurrency), "--out", str(out)]
            if a.judge_model:
                cmd += ["--judge-model", a.judge_model, "--judge-base-url", a.judge_base_url]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                print(f"[:{port}] {tag(layers, alpha)} FAILED:\n{r.stderr[-800:]}", flush=True)

    threads = [threading.Thread(target=worker, args=(port,)) for port in a.ports]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    md = table(out_root, configs)
    (out_root / "sweep_summary.md").write_text(md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
