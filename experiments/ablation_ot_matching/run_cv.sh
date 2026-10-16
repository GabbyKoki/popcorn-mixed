#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

for fold in 0 1 2 3 4; do
  python "$REPO_ROOT/train.py" +experiment=ablation_ot_matching/default data.fold_idx="$fold" data.use_matched_control=false logger.name="ablation_ot_matching__random__fold${fold}" "$@"
done
