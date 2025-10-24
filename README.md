# Project Genesis

## About

Training a 150M parameter language model from scratch on a 10GB text corpus using PyTorch and Vast.ai.

This project implements a transformer-based language model trained from the ground up. The goal is to understand the full pipeline of language model development, from data preparation to training to inference.

## Usage

### Data Preparation

```bash
python data/prepare.py
```

Prepare and preprocess the raw text data for training.

### Training

```bash
python src/train.py
```

Train the language model using the prepared data. Model checkpoints will be saved to the `models/` directory.

### Sampling

```bash
python src/sample.py
```

Generate text using a trained model checkpoint.

### Data Exploration

Open `notebooks/01_data_exploration.ipynb` to explore and analyze the dataset.

## Results

Training results, metrics, and sample outputs will be documented here as the project progresses.
