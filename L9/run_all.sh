#!/usr/bin/env bash
# Graph-mode speedup demo: eager vs triton / mlir / dvm on the deep fusion microbench.
#
# Quick help:  ./run_all.sh -h
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
OUT_DIR="${ROOT}/results"
SHAPE="${SHAPE:-4096 4096}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
MODES=(eager triton mlir dvm)

usage() {
  cat <<EOF
Usage: ./run_all.sh [-h]

What
  Warmup + time avg step for eager / triton / mlir / dvm, compare accuracy to eager,
  then print a speedup table. Clears inductor/triton caches before each backend.

  Workload: deep pointwise (relu/sigmoid/tanh loop x10) — see bench_pointwise.py -h

How
  ./run_all.sh
  SHAPE="2048 2048" WARMUP=5 ITERS=20 ./run_all.sh

Env
  PYTHON   python binary (default: python3)
  SHAPE    "H W" tensor size (default: ${SHAPE})
  WARMUP   warmup iters (default: ${WARMUP})
  ITERS    timed iters (default: ${ITERS})

Where results go  (recreated each run under this script's directory)
  ${OUT_DIR}/
    eager.json  triton.json  mlir.json  dvm.json   # per-backend metrics
    cache_eager/  cache_triton/  cache_mlir/  cache_dvm/
      inductor/   # TORCHINDUCTOR_CACHE_DIR for that run
      triton/     # TRITON_CACHE_DIR for that run
  stdout: per-mode lines + final speedup table

See also
  python3 ${ROOT}/bench_pointwise.py -h
  ${ROOT}/run_tools_demo.sh -h          # dump / accuracy / ANIR / DVM debug tools
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

clear_caches() {
  local cache_root="$1"
  rm -rf "${cache_root}"
  mkdir -p "${cache_root}"
  rm -rf /tmp/torchinductor_* 2>/dev/null || true
  rm -rf "${HOME}/.triton/cache" 2>/dev/null || true
  echo "[cache] cleared -> ${cache_root}"
}

echo "=== graph fusion microbench (deep pointwise) ==="
echo "shape=(${SHAPE}) warmup=${WARMUP} iters=${ITERS}"
echo "results dir: ${OUT_DIR}"
echo "help: ./run_all.sh -h"
echo

for mode in "${MODES[@]}"; do
  cache_root="${OUT_DIR}/cache_${mode}"
  clear_caches "${cache_root}"
  export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor"
  export TRITON_CACHE_DIR="${cache_root}/triton"
  mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

  echo ">>> mode=${mode}"
  ${PY} "${ROOT}/bench_pointwise.py" \
    --mode "${mode}" \
    --shape ${SHAPE} \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --json-out "${OUT_DIR}/${mode}.json"
  echo
done

${PY} - "${OUT_DIR}" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
modes = ["eager", "triton", "mlir", "dvm"]
rows = [json.loads((out / f"{m}.json").read_text()) for m in modes]
eager_ms = rows[0]["avg_ms"]

print("=" * 78)
print(
    f"{'mode':8s} {'avg_ms':>10s} {'speedup':>10s} {'compile_ms':>12s} "
    f"{'max_abs':>12s} {'acc':>6s}  note"
)
print("-" * 78)
for r in rows:
    sp = eager_ms / r["avg_ms"] if r["avg_ms"] > 0 else float("inf")
    note = ""
    if r["mode"] != "eager" and sp < 1.3:
        note = "WARN: ~eager (likely fallback / env issue)"
    print(
        f"{r['mode']:8s} {r['avg_ms']:10.3f} {sp:9.2f}x "
        f"{r['compile_ms']:12.1f} {r['max_abs']:12.3e} "
        f"{'PASS' if r['accuracy_ok'] else 'FAIL':>6s}  {note}"
    )
print("=" * 78)
compiled = [r for r in rows if r["mode"] != "eager"]
sps = [eager_ms / r["avg_ms"] for r in compiled if r["avg_ms"] > 0]
if sps:
    print(
        f"eager baseline: {eager_ms:.3f} ms/iter | "
        f"compiled speedups: min={min(sps):.2f}x  max={max(sps):.2f}x  "
        f"spread={max(sps)/min(sps):.2f}x"
    )
print(f"json metrics: {out}/{{eager,triton,mlir,dvm}}.json")
PY
