#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

modes=(pure mixed)
fusions=(add gated cross_attn)

for mode in "${modes[@]}"; do
  for fusion in "${fusions[@]}"; do
    run_name="ablation_fusion_mode__${mode}__${fusion}__ot"
    echo "Running ${run_name}"
    python "$REPO_ROOT/train.py" +experiment=ablation_fusion_condition_mode/default \
      data.condition_mode="$mode" \
      model.fusion_type="$fusion" \
      logger.name="$run_name" \
      "$@"
  done
done
