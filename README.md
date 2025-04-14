# Granular Ball-based Approximate Borderline Sampling (GBABS)

This repository contains the official implementation of the paper "Approximate Borderline Sampling using Granular-Ball for Classification Tasks". GBABS is a novel boundary sampling algorithm that utilizes granular balls to identify and sample boundary points in classification tasks.

## Method Overview

GBABS leverages the concept of granular balls to perform approximate boundary sampling in data. The algorithm consists of the following steps:

1. Constructing granular balls: Dividing the data space into multiple granular balls, each containing data points of the same class
2. Identifying boundary regions: Finding adjacent balls of different classes along feature dimensions
3. Boundary point sampling: Extracting samples from boundary balls that are close to the decision boundary

This method effectively reduces the training data size while preserving the boundary information crucial for classification decisions.

## Code Structure

The implementation consists of three main files:

- **RD_GBG.py**: Granular ball generation module, implementing the construction and management of granular balls
- **GBABS.py**: Implementation of the granular ball-based boundary sampling approach
- **main.py**: Experiment runner and evaluation script

### RD_GBG.py

This module implements the construction and management of granular balls:

- `GranularBall` class: Represents a single granular ball with attributes like data points, center, radius, and label
- `GranularBallManager` class: Manages the generation, manipulation, and retrieval of granular balls
- Helper functions: Distance calculation, outlier detection, ball generation, etc.

### GBABS.py

Implements the granular ball-based approximate boundary sampling method:

- GBABS class: Main implementation class containing the core logic for boundary point sampling
- `bound_sampling`: Main sampling method that analyzes boundary relationships between granular balls of different classes
- `extract_boundary_samples`: Extracts boundary samples from balls based on specific feature dimensions

### main.py

Experiment and evaluation script:

- Data loading and preprocessing
- Experiment setup (cross-validation, parameter settings)
- Comparison with other sampling methods (random sampling, SMOTE variants, other boundary sampling methods)
- Result statistics and output

## Usage

### Requirements

```
numpy
pandas
scikit-learn
imbalanced-learn
xgboost
lightgbm
```

### Basic Usage

```python
# Load your dataset
data = load_your_dataset()

# Create GBABS instance
gbabs = GBABS.GBABS(data, rho=5)  # rho is the density parameter for granular ball construction

# Perform boundary sampling
boundary_samples = gbabs.bound_sampling()

# Train a classifier using the sampled data
X = boundary_samples[:, 1:]  # Features
y = boundary_samples[:, 0]   # Labels
classifier = train_your_classifier(X, y)
```

### Running Experiments

You can run the experiments from the paper by executing main.py:

```bash
python main.py
```

Parameters can be adjusted in main.py, including:
- `rho`: Density parameter for granular ball construction
- `Noise_ratio`: Noise ratio in the data
- `repetitions`: Number of experiment repetitions
- `baseline`: Classifier type to use

## Experimental Results

The comparison between GBABS and other sampling methods is based on the following metrics:
- Sampling rate
- Accuracy
- Geometric mean (G-mean)

Experiments show that GBABS can achieve comparable or even better classification performance than using the full dataset while maintaining a lower sampling rate on most datasets.

## Citation

If you use the GBABS algorithm or this code in your research, please cite the original paper:

```
@article{xie2023approximate,
  title={Approximate Borderline Sampling using Granular-Ball for Classification Tasks},
  author={Xie, Qin and Tse, Cheryl and Feng, Quan and Yang, Yingqin},
  journal={[Journal Name]},
  year={2023}
}
```
