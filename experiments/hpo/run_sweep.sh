#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sweep_config_name> [hydra overrides...]"
  echo "Example: $0 hyper"
  exit 1
fi

sweep_cfg="$1"
shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

python "$REPO_ROOT/train.py" -m +experiment=hpo/default +sweep="$sweep_cfg" "$@"
