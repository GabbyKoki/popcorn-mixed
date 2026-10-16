#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

python "$REPO_ROOT/compare_modes.py" logger.name="comparison_prediction_mode__absolute_vs_residual" "$@"
