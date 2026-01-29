# SimpleGPT - Text Generation with Transformer

A simple GPT-like transformer model implementation for text generation, built from scratch using PyTorch. This project demonstrates the core concepts of transformer architecture including multi-head attention, feed-forward networks, and positional encoding.

## Features

- **Byte Pair Encoding (BPE)** tokenization - subword tokenization for better efficiency
- Transformer architecture with multi-head attention
- Training with TensorBoard logging
- Text generation with temperature sampling
- Simple and educational implementation

## Project Structure

```
.
├── model.py           # Transformer model implementation (SimpleGPT)
├── tokenizer.py       # BPE tokenization and dataset
├── bpe_tokenizer.json # Saved BPE tokenizer (created after first run)
├── prepare_data.py    # Data cleaning and preparation
├── train.py          # Training script
├── generate.py       # Text generation script
├── data_set.txt      # Input text data
├── cleaned_dataset.txt # Processed text data
└── model_epoch_*.pth  # Saved model checkpoints
```

## Installation

1. Clone or download this repository

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Your Data

Place your text data in `data_set.txt` or modify `prepare_data.py` to load from a different source. The script will:
- Clean the text (lowercase, remove punctuation, normalize whitespace)
- Save the cleaned data to `cleaned_dataset.txt`

```bash
python prepare_data.py
```

### 2. Train the Model

Train the model on your dataset:

```bash
python train.py
```

The training script will:
- Load and tokenize the cleaned dataset using BPE (or load existing tokenizer)
- Train the BPE tokenizer if `bpe_tokenizer.json` doesn't exist (first run only)
- Train the model for 5000 epochs (configurable)
- Save model checkpoints every 500 epochs
- Log training loss to TensorBoard

To view training progress in TensorBoard:
```bash
tensorboard --logdir=runs
```

### 3. Generate Text

After training, generate text using a trained model:

```bash
python generate.py
```

**Note:** Make sure to update the model checkpoint path in `generate.py` (line 24) to match your latest saved model (e.g., `model_epoch_4500.pth`).

You can customize generation parameters:
- `max_new_tokens`: Number of tokens to generate (default: 200)
- `temperature`: Sampling temperature - higher values make output more random (default: 0.8)

## Model Architecture

The model implements a simplified GPT architecture:

- **Token Embedding**: Maps BPE tokens to dense vectors
- **Positional Embedding**: Adds positional information
- **Transformer Blocks**: Stack of self-attention and feed-forward layers
- **Output Head**: Linear layer for next-token prediction
- **Weight Tying**: Shares weights between embedding and output layers

### Hyperparameters (configurable in `model.py`):

- `n_embd`: Embedding dimension (default: 128)
- `n_head`: Number of attention heads (default: 4)
- `n_layer`: Number of transformer blocks (default: 4)
- `block_size`: Maximum sequence length (default: 128)

## Requirements

- Python 3.7+
- PyTorch 2.0+
- TensorBoard (for training visualization)

## Tokenization

This project uses **Byte Pair Encoding (BPE)** for tokenization:

- **BPE** is a subword tokenization method that starts with character-level vocabulary and iteratively merges the most frequent pairs
- More efficient than character-level (smaller sequences)
- Better than word-level (handles out-of-vocabulary words)
- The tokenizer is automatically trained on first run and saved to `bpe_tokenizer.json`
- Default vocabulary size: 1000 tokens (configurable in `tokenizer.py`)

## Notes

- BPE tokenization provides a good balance between character and word-level approaches
- Training time depends on dataset size and hardware (CPU/GPU)
- Model checkpoints are saved periodically during training
- The model automatically uses CUDA if available, otherwise falls back to CPU
- BPE tokenizer is saved after training and reused on subsequent runs

## License

This is an educational implementation for learning transformer architectures.
