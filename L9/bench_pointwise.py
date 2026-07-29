#!/usr/bin/env python3
"""Deep pointwise fusion microbench: eager vs inductor (triton / mlir / dvm).

Quick help
----------
  python3 bench_pointwise.py -h
  python3 bench_pointwise.py --mode eager
  python3 bench_pointwise.py --mode triton --json-out /tmp/triton.json

Usually prefer the orchestrators (they clear caches & summarize):
  ./run_all.sh              # speedup vs eager (all backends)
  ./run_tools_demo.sh list  # Ascend dump / accuracy / debug tools

What it measures
----------------
A deep mul/add/relu/sigmoid/tanh chain (~50+ pointwise ops). Eager pays many
kernel launches; compile should fuse into ~1 kernel → large speedup if fusion
works on the backend.

Outputs
-------
  stdout : one line with avg_ms / compile_ms / max_abs / accuracy PASS|FAIL
  --json-out PATH : same metrics as JSON (created by run_all.sh / run_tools_demo.sh)

Notes
-----
  triton -> env TORCHINDUCTOR_NPU_BACKEND=default
  mlir   -> TORCHINDUCTOR_NPU_BACKEND=mlir
  dvm    -> TORCHINDUCTOR_NPU_BACKEND=dvm
"""

from __future__ import annotations

import argparse
import json
import os
import time

# User-facing name -> TORCHINDUCTOR_NPU_BACKEND (triton == default).
_BACKEND_ENV = {"triton": "default", "mlir": "mlir", "dvm": "dvm"}


def parse_args():
    p = argparse.ArgumentParser(
        description="NPU deep pointwise fusion microbench (eager / triton / mlir / dvm).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --mode eager
  %(prog)s --mode triton --shape 2048 2048
  %(prog)s --mode mlir --json-out results/mlir.json

workload (PPT / teaching snippet):
  y = x * w + b
  for _ in range(10):
      y = relu(y); y = y * 1.01 + 0.01; y = sigmoid(y)
      y = y * w - b * 0.1; y = tanh(y)

results:
  stdout line:
    [mode] shape=... avg=X ms/iter compile=Y ms max_abs=... accuracy=PASS|FAIL
  --json-out writes: mode, avg_ms, compile_ms, max_abs, max_rel, accuracy_ok, ...

see also:
  ./run_all.sh            -> results/{eager,triton,mlir,dvm}.json + speedup table
  ./run_tools_demo.sh -h  -> tool_runs/<tool>_<mode>_<ts>/ (dumps & logs)
""".rstrip(),
    )
    p.add_argument(
        "--mode",
        choices=["eager", "triton", "mlir", "dvm"],
        required=True,
        help="eager baseline, or inductor backend (triton=default NPU backend)",
    )
    p.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=[4096, 4096],
        metavar=("H", "W"),
        help="tensor shape (default: 4096 4096; use smaller if OOM)",
    )
    p.add_argument("--warmup", type=int, default=10, help="warmup iters (not timed)")
    p.add_argument("--iters", type=int, default=50, help="timed iters for avg_ms")
    p.add_argument("--atol", type=float, default=1e-3, help="allclose atol vs eager ref")
    p.add_argument("--rtol", type=float, default=1e-3, help="allclose rtol vs eager ref")
    p.add_argument(
        "--json-out",
        default="",
        metavar="PATH",
        help="optional JSON metrics path (dir created if needed)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode != "eager":
        os.environ["TORCHINDUCTOR_NPU_BACKEND"] = _BACKEND_ENV[args.mode]

    import torch
    import torch_npu  # noqa: F401

    def deep_pointwise(x, w, b):
        # Eager: many launches + HBM traffic; compile: typically 1 fused kernel.
        y = x * w + b
        for _ in range(10):
            y = torch.relu(y)
            y = y * 1.01 + 0.01
            y = torch.sigmoid(y)
            y = y * w - b * 0.1
            y = torch.tanh(y)
        return y

    def sync():
        torch.npu.synchronize()

    def bench(fn, inputs, warmup, iters):
        for _ in range(warmup):
            fn(*inputs)
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(*inputs)
        sync()
        return out, (time.perf_counter() - t0) / iters * 1e3

    device = "npu"
    shape = tuple(args.shape)
    inputs = (
        torch.randn(*shape, device=device),
        torch.randn(*shape, device=device),
        torch.randn(*shape, device=device),
    )

    ref, _ = bench(deep_pointwise, inputs, max(args.warmup // 2, 1), 1)

    compile_ms = 0.0
    if args.mode == "eager":
        fn = deep_pointwise
    else:
        fn = torch.compile(deep_pointwise, backend="inductor", dynamic=False)
        t0 = time.perf_counter()
        _ = fn(*inputs)
        sync()
        compile_ms = (time.perf_counter() - t0) * 1e3

    out, avg_ms = bench(fn, inputs, args.warmup, args.iters)
    max_abs = (out - ref).abs().max().item()
    max_rel = ((out - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
    ok = torch.allclose(out, ref, atol=args.atol, rtol=args.rtol)

    result = {
        "mode": args.mode,
        "shape": list(shape),
        "warmup": args.warmup,
        "iters": args.iters,
        "compile_ms": round(compile_ms, 3),
        "avg_ms": round(avg_ms, 3),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "accuracy_ok": bool(ok),
        "atol": args.atol,
        "rtol": args.rtol,
    }
    print(
        f"[{args.mode:7s}] shape={shape}  avg={avg_ms:.3f} ms/iter  "
        f"compile={compile_ms:.1f} ms  max_abs={max_abs:.3e}  "
        f"max_rel={max_rel:.3e}  accuracy={'PASS' if ok else 'FAIL'}"
    )
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[json] wrote {os.path.abspath(args.json_out)}")


if __name__ == "__main__":
    main()
