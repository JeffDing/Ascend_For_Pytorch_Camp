#!/usr/bin/env bash
# Opt-in demos of Ascend inductor dump / accuracy / debug tools.
#
# Quick help:  ./run_tools_demo.sh -h
# Default with no args: print catalog (tools stay OFF until you pick one).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
# Tools add overhead; keep shape smaller for demos (same deep loop as run_all).
SHAPE="${SHAPE:-1024 1024}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-5}"
MODE="triton"
TOOL=""
OUT_ROOT="${ROOT}/tool_runs"

usage() {
  cat <<EOF
Usage: ./run_tools_demo.sh <tool> [--mode triton|mlir|dvm] [options]

What
  Run bench_pointwise.py (deep pointwise fusion) once with selected debug env vars.
  Artifacts are isolated under tool_runs/ (not mixed with ./run_all.sh results/).

How
  ./run_tools_demo.sh -h
  ./run_tools_demo.sh list
  ./run_tools_demo.sh torch_compile_debug --mode triton
  ./run_tools_demo.sh dump_fx --mode triton
  ./run_tools_demo.sh check_accuracy --mode mlir
  ./run_tools_demo.sh anir_debug --mode mlir
  ./run_tools_demo.sh dvm_debug --mode dvm
  ./run_tools_demo.sh combo_triton

Options / env
  --mode triton|mlir|dvm   (default: triton; combo_* forces its backend)
  --shape H W              (default: ${SHAPE})
  --warmup N --iters N
  PYTHON SHAPE WARMUP ITERS   env overrides also work

Where results go
  ${OUT_ROOT}/<tool>_<mode>_<timestamp>/
    result.json          bench metrics
    run.log              full stdout/stderr
    cache/
      inductor/          TORCHINDUCTOR_CACHE_DIR (FX dumps live here when enabled)
      triton/            TRITON_CACHE_DIR
    torch_compile_debug/ TORCH_COMPILE_DEBUG=1 traces (cwd = this run dir)
    artifacts/
      anir_debug/        ANIR_DEBUG=1 logs
      torch_mlir/        ANIR_TORCH_MLIR_DUMP

  After each run the script prints "=== where to look ===" with concrete paths.

Tools (default OFF; pick one)
  list                 Print this help only.
  torch_compile_debug  TORCH_COMPILE_DEBUG=1
                       -> <run_dir>/torch_compile_debug/run_*/
  torch_logs           TORCH_LOGS=output_code,graph_code  (mostly in run.log)
  dump_fx              INDUCTOR_ASCEND_DUMP_FX_GRAPH=1
                       -> cache/inductor/traced_fx_graph_cache/**/*.py (+ data.pth)
  check_accuracy       INDUCTOR_ASCEND_CHECK_ACCURACY=1 (also enables dump_fx)
                       compare fused kernel vs FX; optional
                       INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL=rtol=1e-3,atol=1e-3
  ascend_log           INDUCTOR_ASCEND_LOG_LEVEL=DEBUG  -> run.log
  anir_debug           ANIR_DEBUG=1 (+ FALLBACK/RUNTIME dumps) -> artifacts/anir_debug/
  anir_mlir_dump       ANIR_TORCH_MLIR_DUMP -> artifacts/torch_mlir/
  dvm_debug            INDUCTOR_DVM_DEBUG_MODE=1  (extra dump on DVM sync fail)
  dvm_fx_test          INDUCTOR_DVM_DUMP_FX_TEST=1
  fallback_warn        INDUCTOR_ASCEND_FALLBACK_WARNING=1 / ANIR_FALLBACK_WARNING=1
  no_autofallback      INDUCTOR_ASCEND_AUTO_FALLBACK=0  (fail instead of silent fallback)
  combo_triton         torch_compile_debug + dump_fx + check_accuracy
  combo_mlir           anir_debug + check_accuracy + dump_fx
  combo_dvm            dvm_debug + dvm_fx_test + dump_fx

Note: DUMP_FX / CHECK_ACCURACY are not compatible with aclgraph.

See also
  ./run_all.sh -h                         # speedup table -> ./results/
  python3 ${ROOT}/bench_pointwise.py -h   # single-backend microbench
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --shape) SHAPE="$2 $3"; shift 3 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      if [[ -z "${TOOL}" ]]; then TOOL="$1"; shift
      else echo "Unknown arg: $1"; usage; exit 1; fi
      ;;
  esac
done

if [[ -z "${TOOL}" || "${TOOL}" == "list" || "${TOOL}" == "help" ]]; then
  usage
  exit 0
fi

case "${MODE}" in
  triton|mlir|dvm) ;;
  *) echo "mode must be triton|mlir|dvm (eager has no inductor tools)"; exit 1 ;;
esac

RUN_DIR="${OUT_ROOT}/${TOOL}_${MODE}_$(date +%Y%m%d_%H%M%S)"
CACHE_DIR="${RUN_DIR}/cache"
mkdir -p "${CACHE_DIR}/inductor" "${CACHE_DIR}/triton" "${RUN_DIR}/artifacts"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/inductor"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton"
mkdir -p "${CACHE_DIR}/inductor" "${CACHE_DIR}/triton"

apply_tool_env() {
  case "$1" in
    torch_compile_debug)
      export TORCH_COMPILE_DEBUG=1
      ;;
    torch_logs)
      export TORCH_LOGS="${TORCH_LOGS:-output_code,graph_code}"
      ;;
    dump_fx)
      export INDUCTOR_ASCEND_DUMP_FX_GRAPH=1
      ;;
    check_accuracy)
      export INDUCTOR_ASCEND_CHECK_ACCURACY=1
      export INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL="${INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL:-rtol=1e-3,atol=1e-3}"
      ;;
    ascend_log)
      export INDUCTOR_ASCEND_LOG_LEVEL=DEBUG
      ;;
    anir_debug)
      export ANIR_DEBUG=1
      export ANIR_DEBUG_DIR="${RUN_DIR}/artifacts/anir_debug"
      export ANIR_FALLBACK_DUMP=1
      export ANIR_RUNTIME_ERROR_DUMP=1
      mkdir -p "${ANIR_DEBUG_DIR}"
      ;;
    anir_mlir_dump)
      export ANIR_TORCH_MLIR_DUMP="${RUN_DIR}/artifacts/torch_mlir"
      mkdir -p "${ANIR_TORCH_MLIR_DUMP}"
      ;;
    dvm_debug)
      export INDUCTOR_DVM_DEBUG_MODE=1
      ;;
    dvm_fx_test)
      export INDUCTOR_DVM_DUMP_FX_TEST=1
      ;;
    fallback_warn)
      export INDUCTOR_ASCEND_FALLBACK_WARNING=1
      export ANIR_FALLBACK_WARNING=1
      ;;
    no_autofallback)
      export INDUCTOR_ASCEND_AUTO_FALLBACK=0
      ;;
    combo_triton)
      export TORCH_COMPILE_DEBUG=1
      export INDUCTOR_ASCEND_DUMP_FX_GRAPH=1
      export INDUCTOR_ASCEND_CHECK_ACCURACY=1
      export INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL="${INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL:-rtol=1e-3,atol=1e-3}"
      export INDUCTOR_ASCEND_LOG_LEVEL=INFO
      MODE=triton
      ;;
    combo_mlir)
      export ANIR_DEBUG=1
      export ANIR_DEBUG_DIR="${RUN_DIR}/artifacts/anir_debug"
      export ANIR_FALLBACK_DUMP=1
      export ANIR_RUNTIME_ERROR_DUMP=1
      export INDUCTOR_ASCEND_DUMP_FX_GRAPH=1
      export INDUCTOR_ASCEND_CHECK_ACCURACY=1
      mkdir -p "${ANIR_DEBUG_DIR}"
      MODE=mlir
      ;;
    combo_dvm)
      export INDUCTOR_DVM_DEBUG_MODE=1
      export INDUCTOR_DVM_DUMP_FX_TEST=1
      export INDUCTOR_ASCEND_DUMP_FX_GRAPH=1
      MODE=dvm
      ;;
    *)
      echo "Unknown tool: $1"; usage; exit 1
      ;;
  esac
}

apply_tool_env "${TOOL}"

LOG="${RUN_DIR}/run.log"
echo "=== tools demo (deep pointwise) ==="
echo "tool=${TOOL}  mode=${MODE}  shape=(${SHAPE})"
echo "run_dir=${RUN_DIR}"
echo "log=${LOG}"
echo "help: ./run_tools_demo.sh -h"
echo

# Run under RUN_DIR so TORCH_COMPILE_DEBUG dumps land here (cwd-relative).
set +e
(
  cd "${RUN_DIR}"
  ${PY} "${ROOT}/bench_pointwise.py" \
    --mode "${MODE}" \
    --shape ${SHAPE} \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --json-out "${RUN_DIR}/result.json"
) 2>&1 | tee "${LOG}"
rc=${PIPESTATUS[0]}
set -e

echo
echo "=== where to look ==="
echo "run_dir     : ${RUN_DIR}"
echo "result json : ${RUN_DIR}/result.json"
echo "full log    : ${LOG}"

if [[ -d "${RUN_DIR}/torch_compile_debug" ]]; then
  echo "torch_compile_debug:"
  ls -1dt "${RUN_DIR}/torch_compile_debug"/run_* 2>/dev/null | head -5 || true
fi
if [[ -d "${TORCHINDUCTOR_CACHE_DIR}/traced_fx_graph_cache" ]]; then
  echo "dump_fx / check_accuracy FX graphs:"
  find "${TORCHINDUCTOR_CACHE_DIR}/traced_fx_graph_cache" \( -name '*.py' -o -name 'data.pth' \) 2>/dev/null | head -30 || true
fi
if [[ -n "${ANIR_DEBUG_DIR:-}" && -d "${ANIR_DEBUG_DIR}" ]]; then
  echo "anir_debug:"
  find "${ANIR_DEBUG_DIR}" -type f 2>/dev/null | head -20 || true
fi
if [[ -n "${ANIR_TORCH_MLIR_DUMP:-}" && -d "${ANIR_TORCH_MLIR_DUMP}" ]]; then
  echo "torch_mlir dump:"
  find "${ANIR_TORCH_MLIR_DUMP}" -type f 2>/dev/null | head -20 || true
fi
echo "other artifacts under run_dir:"
find "${RUN_DIR}" -type f \( -name '*.py' -o -name '*.mlir' -o -name '*.log' -o -name '*.pth' -o -name '*.json' \) 2>/dev/null | head -40 || true

echo
if [[ ${rc} -ne 0 ]]; then
  echo "bench exited with code ${rc} (some tools force fail-on-fallback / accuracy miss)."
  exit "${rc}"
fi
echo "done. Re-run with another tool, or ./run_all.sh for speedup table."
