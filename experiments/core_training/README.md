# Core Training

This family covers standard model training with the main `train.py` entrypoint.

## Axes

- `data.fold_idx`
- `data.condition_mode` (`mixed` or `pure`)
- `data.use_matched_control` (`true` or `false`)
- `model.fusion_type` (`add`, `gated`, `cross_attn`)

## Commands

Single run:

```bash
bash experiments/core_training/run.sh
```

With overrides:

```bash
bash experiments/core_training/run.sh data.fold_idx=1 data.condition_mode=pure model.fusion_type=cross_attn
```

Equivalent direct command:

```bash
python train.py +experiment=core_training/default data.fold_idx=0
```
