# Learning without Forgetting (LwF) Implementation

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.8%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

This repository implements the Learning without Forgetting (LwF) method for continual learning using PyTorch. The implementation demonstrates how LwF helps mitigate catastrophic forgetting when training sequentially on MNIST and Fashion-MNIST datasets.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Overview

Learning without Forgetting (LwF) is a continual learning approach that enables neural networks to learn new tasks while retaining performance on previously learned tasks. This implementation:

- Trains a model on MNIST (Task 1)
- Then trains the same model on Fashion-MNIST (Task 2)
- Compares standard training (which suffers from catastrophic forgetting) with LwF training
- Provides visualizations of the forgetting metrics

## Features

- Complete implementation of LwF method
- Comparison with standard training approach
- Modular code structure for easy extension
- Training metrics tracking and visualization
- CLI interface for configuration
- Model checkpointing
- Reproducibility through random seed control

## Installation

1. Clone the repository:

```bash
git clone https://github.com/thealper2/pytorch-lwf.git
cd pytorch-lwf
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage`

### Running the experiment

```bash
python3 main.py \
  --learning-rate 0.001 \
  --batch-size 128 \
  --epochs-task1 5 \
  --epochs-task2 5 \
  --temperature 2.0 \
  --lambda-old 1.0 \
  --seed 42 \
  --save-dir "model_checkpoints" \
  --use-cuda True \
  --log-interval 100
```

### Command Line Options

```bash
 Usage: main.py [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --learning-rate                     FLOAT    Learning rate for optimizer [default: 0.001]                │
│ --batch-size                        INTEGER  Batch size for training [default: 128]                      │
│ --epochs-task1                      INTEGER  Number of epochs for Task 1 (MNIST) [default: 5]            │
│ --epochs-task2                      INTEGER  Number of epochs for Task 2 (Fashion-MNIST) [default: 5]    │
│ --temperature                       FLOAT    Temperature parameter for LwF knowledge distillation        │
│                                              [default: 2.0]                                              │
│ --lambda-old                        FLOAT    Weight for old task loss in LwF [default: 1.0]              │
│ --seed                              INTEGER  Random seed for reproducibility [default: 42]               │
│ --save-dir                          TEXT     Directory to save model checkpoints                         │
│                                              [default: model_checkpoints]                                │
│ --use-cuda         --no-use-cuda             Use GPU if available [default: use-cuda]                    │
│ --log-interval                      INTEGER  How many batches to wait before logging training status     │
│                                              [default: 100]                                              │
│ --help                                       Show this message and exit.                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (git checkout -b feature-branch)
3. Commit your changes (git commit -am 'Add new feature')
4. Push to the branch (git push origin feature-branch)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.