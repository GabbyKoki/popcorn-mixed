# Pop-Corn

Predict cell-state composition after genetic perturbation from control-cell and perturbation embeddings.

## Setup

```bash
pip install -r requirements.txt
wandb login
```

## Training

Prepare an AnnData file and matching index following `conf/data/all.yaml`, then run:

```bash
python train.py data=all data.processed_data_dir=/path/to/data
# Five-fold cross-validation
python train.py -m data=all data.fold_idx=0,1,2,3,4 data.processed_data_dir=/path/to/data
```

Model and training settings are in `conf/`. Predictions and metrics are saved in each run's Hydra output directory. Data and experiment outputs are not included.

MIT license.
