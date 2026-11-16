export NEXUS_WORKDIR="${NEXUS_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export NEXUS_CACHE_ROOT="${NEXUS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/nexus}"

export XDG_CACHE_HOME="$NEXUS_CACHE_ROOT"
export CONDA_PKGS_DIRS="$NEXUS_CACHE_ROOT/conda/pkgs"
export PIP_CACHE_DIR="$NEXUS_CACHE_ROOT/pip"
export HF_HOME="$NEXUS_CACHE_ROOT/huggingface"
export TORCH_HOME="$NEXUS_CACHE_ROOT/torch"
export RCLONE_CACHE_DIR="$NEXUS_CACHE_ROOT/rclone"

export NUPLAN_DEVKIT_PATH="$NEXUS_WORKDIR/Nexus/third_party/nuplan-devkit"
