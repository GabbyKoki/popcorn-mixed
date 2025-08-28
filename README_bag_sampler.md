# Bag Sampler Usage Guide

## Overview

The new `BagSampler` class provides flexible sampling strategies for creating training mini-bags. This allows for better training diversity and more efficient batch processing.

## Available Sampling Strategies

### 1. Bootstrap Sampling (`BootstrapBagSampler`)
- **Strategy**: Sampling with replacement
- **Use case**: Default option, provides good diversity
- **Advantages**: Can generate many diverse bags, handles class imbalance well
- **Configuration**: `sampler_type: "bootstrap"`

### 2. Stratified Sampling (`StratifiedBagSampler`)
- **Strategy**: Maintains original class proportions in each bag
- **Use case**: When you want consistent class distribution across bags
- **Advantages**: Ensures balanced representation of all classes
- **Configuration**: `sampler_type: "stratified"`

### 3. Uniform Sampling (`UniformBagSampler`)
- **Strategy**: Sampling without replacement
- **Use case**: When you want non-overlapping samples
- **Advantages**: Ensures no duplicate cells within each bag
- **Configuration**: `sampler_type: "uniform"`

## Configuration Examples

### Basic Usage
```bash
# Use default bootstrap sampling
python train.py

# Use stratified sampling
python train.py training=stratified

# Use uniform sampling
python train.py training=uniform
```

### Custom Parameters
```bash
# Override specific parameters
python train.py training.sampler_type="stratified" training.n_bags=40 training.sample_ratio=0.4

# Use with hyperparameter sweeps
python train.py -m --config-name=sweep/hyper training.sampler_type="bootstrap","stratified","uniform"
```

## Configuration Files

### Default Configuration (`conf/training/default.yaml`)
```yaml
n_bags: 50
sample_ratio: 0.3
sampler_type: "bootstrap"
```

### Stratified Configuration (`conf/training/stratified.yaml`)
```yaml
sampler_type: "stratified"
sample_ratio: 0.4
n_bags: 30
```

### Uniform Configuration (`conf/training/uniform.yaml`)
```yaml
sampler_type: "uniform"
sample_ratio: 0.5
n_bags: 20
```

## Implementation Details

### Key Features
1. **Modular Design**: Easy to add new sampling strategies
2. **Reproducible**: All samplers use random seeds for consistency
3. **Training-Only**: Only affects training bags, validation/test remain unchanged
4. **Bag Metadata**: Each bag includes a unique `bag_id` for tracking

### Code Structure
```
src/
├── bag_sampler.py      # New BagSampler classes
├── pert_bags.py        # Updated to use BagSampler
├── dataset.py          # Updated to pass sampler_type
└── ...
```

## Performance Considerations

- **Bootstrap**: Fastest, good for large datasets
- **Stratified**: Slightly slower due to class balancing, best for imbalanced data
- **Uniform**: Fastest per bag, but limited by dataset size (no replacement)

## Best Practices

1. **Start with Bootstrap**: Good default for most cases
2. **Use Stratified**: When class imbalance is a concern
3. **Use Uniform**: When you want maximum diversity without overlap
4. **Adjust `n_bags`**: More bags = more training diversity but longer epochs
5. **Adjust `sample_ratio`**: Lower ratios = faster training, higher ratios = more data per bag


