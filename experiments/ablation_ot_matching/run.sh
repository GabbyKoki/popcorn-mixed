#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

echo "OT ablation: matched vs random control"
python "$REPO_ROOT/train.py" +experiment=ablation_ot_matching/default data.use_matched_control=true  logger.name="ablation_ot_matching__matched" "$@"
python "$REPO_ROOT/train.py" +experiment=ablation_ot_matching/default data.use_matched_control=false logger.name="ablation_ot_matching__random" "$@"
