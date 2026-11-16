#!/usr/bin/env bash
# Run the real 3DGS backend with project-local Python/native dependencies.
set -euo pipefail
bridge_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
bridge_python="$bridge_root/artifacts/nexus_3dgs/venv/bin/python"
bridge_native="$bridge_root/artifacts/nexus_3dgs/native/usr/lib/x86_64-linux-gnu"
if [[ ! -x "$bridge_python" ]]; then
  echo "Missing 3DGS environment: $bridge_python" >&2
  exit 1
fi
case "${1:-}" in
  train) bridge_entry="$bridge_root/Omniscenegraph/tools/train.py" ;;
  eval) bridge_entry="$bridge_root/Omniscenegraph/tools/eval.py" ;;
  render) bridge_entry="$bridge_root/Omniscenegraph/sim_render/cam/render_nexus.py" ;;
  render-jobs) bridge_entry="$bridge_root/Omniscenegraph/sim_render/cam/render_nexus_jobs.py" ;;
  probe) bridge_entry="$bridge_root/scripts/pipeline/probe_nexus_3dgs_runtime.py" ;;
  verify) bridge_entry="$bridge_root/scripts/pipeline/verify_nexus_3dgs_outputs.py" ;;
  script) shift; bridge_entry="${1:?Absolute Python script path required}" ;;
  *) echo "Usage: $0 {probe|train|eval|render|render-jobs|verify|script} [arguments; use absolute input/output paths]" >&2; exit 2 ;;
esac
shift
export PYTHONPATH="$bridge_root/Omniscenegraph:$bridge_root${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$bridge_native${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_HOME="$bridge_root/artifacts/nexus_3dgs/cache/torch"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/nexus-3dgs-torch-extensions}"
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
cd "$bridge_root/Omniscenegraph"
exec "$bridge_python" "$bridge_entry" "$@"
