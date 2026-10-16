#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# CV is a Hydra multirun by default.
python "$REPO_ROOT/train.py" -m +experiment=cv/default "$@"
