# SimpleGPT: Complete A-to-Z Guide

## 📚 Table of Contents
1. [Overview](#overview)
2. [Project Architecture](#project-architecture)
3. [File-by-File Breakdown](#file-by-file-breakdown)
4. [Key Concepts Explained](#key-concepts-explained)
5. [How Everything Works Together](#how-everything-works-together)
6. [Step-by-Step Workflow](#step-by-step-workflow)

---

## 🎯 Overview

This is a **SimpleGPT** project - a simplified implementation of a GPT (Generative Pre-trained Transformer) model built from scratch using PyTorch. It's designed for educational purposes to understand how transformer-based language models work.

**What it does:** Trains a neural network to generate text character-by-character, learning patterns from your input data.

---

## 🏗️ Project Architecture

```
Input Text → Data Cleaning → Tokenization → Training → Model → Text Generation
```

### The Pipeline:
1. **Data Preparation** (`prepare_data.py`) - Cleans raw text
2. **Tokenization** (`tokenizer.py`) - Converts text to numbers
3. **Model Definition** (`model.py`) - Neural network architecture
4. **Training** (`train.py`) - Teaches the model
5. **Generation** (`generate.py`) - Creates new text

---

## 📁 File-by-File Breakdown

### 1. `requirements.txt`
**Purpose:** Lists all Python packages needed to run this project.

**Contents:**
- `torch>=2.0.0` - PyTorch library for neural networks
- `tensorboard>=2.14.0` - Visualization tool for training progress

**How to use:**
```bash
pip install -r requirements.txt
```

---

### 2. `data_set.txt`
**Purpose:** Your raw input text file. This is where you put the text you want the model to learn from.

**Example content:** Any text - stories, articles, code, etc.

---

### 3. `prepare_data.py`
**Purpose:** Cleans and preprocesses raw text data.

**What it does:**
- Reads `data_set.txt`
- Converts text to lowercase
- Removes punctuation
- Normalizes whitespace (multiple spaces → single space)
- Saves cleaned text to `cleaned_dataset.txt`

**Key Functions:**
```python
# Line 14: Normalizes whitespace
text = re.sub(r'\s+', ' ', text.lower())

# Line 15: Removes punctuation
text = re.sub(r'[^\w\s]', '', text)
```

**Why clean data?** 
- Consistent format helps the model learn better
- Reduces vocabulary size (no uppercase/lowercase duplicates)
- Removes noise that might confuse the model

---

### 4. `tokenizer.py`
**Purpose:** Converts text into numbers that the neural network can understand.

**Key Components:**

#### `TextDataset` Class (Lines 4-21)
- **Character-level tokenization:** Each character becomes a number
- **Vocabulary:** Creates a mapping between characters and numbers
  - `stoi` (string-to-index): `{'a': 0, 'b': 1, ...}`
  - `itos` (index-to-string): `{0: 'a', 1: 'b', ...}`
- **Data encoding:** Converts entire text into a tensor of numbers
- **Sequence chunks:** Splits text into training examples

**How it works:**
```python
# Example: "hello" becomes [7, 4, 11, 11, 14] (if h=7, e=4, l=11, o=14)
self.data = torch.tensor([self.stoi[ch] for ch in text])
```

**DataLoader (Line 29):**
- Batches data for efficient training
- Shuffles data to prevent overfitting
- `batch_size=32`: Processes 32 sequences at once

**Why character-level?**
- Works with any text (no need for a predefined vocabulary)
- Good for smaller datasets
- Can handle any character (emojis, special symbols, etc.)

---

### 5. `model.py`
**Purpose:** Defines the neural network architecture - the "brain" of the system.

#### **SimpleGPT Class** (Main Model)
**Architecture Components:**

1. **Token Embedding** (Line 9)
   - Converts character indices → dense vectors
   - Example: Character 'a' → [0.2, -0.5, 0.8, ...] (128-dimensional)

2. **Positional Embedding** (Line 10)
   - Adds position information to each character
   - Why? "cat" vs "tac" - same characters, different meaning
   - Each position gets a unique embedding

3. **Transformer Blocks** (Line 14)
   - Stack of attention + feed-forward layers
   - Default: 4 blocks (configurable)

4. **Output Head** (Line 17)
   - Final layer that predicts next character
   - Outputs probability distribution over all characters

5. **Weight Tying** (Line 20)
   - Shares weights between embedding and output layers
   - Reduces parameters, improves efficiency

**Forward Pass Flow:**
```
Input tokens → Embeddings → Position encoding → Transformer blocks → Output probabilities
```

#### **TransformerBlock Class** (Lines 38-49)
**Structure:**
- **Self-Attention** (`MultiHeadAttention`)
- **Feed-Forward Network**
- **Layer Normalization** (before each sub-layer)
- **Residual Connections** (skip connections)

**Why this structure?**
- Attention: Model learns which characters to focus on
- Feed-forward: Adds non-linearity and processing power
- Residuals: Helps with gradient flow during training
- Layer norm: Stabilizes training

#### **MultiHeadAttention Class** (Lines 51-75)
**What is Attention?**
- Mechanism that lets the model "look" at all previous characters
- Calculates relationships between characters
- Example: In "the cat", attention might link "the" and "cat"

**How it works:**
1. **Query, Key, Value** (Line 57): Creates Q, K, V from input
2. **Attention Scores** (Line 67): `Q @ K^T` - measures similarity
3. **Causal Masking** (Line 68): Prevents looking at future tokens
4. **Softmax** (Line 69): Converts scores to probabilities
5. **Weighted Sum** (Line 72): Combines values based on attention

**Multi-Head:** Runs attention multiple times in parallel, each focusing on different relationships.

#### **FeedForward Class** (Lines 77-88)
**Structure:**
- Expands dimension: 128 → 512 → 128
- Uses GELU activation (smooth ReLU alternative)
- Adds dropout for regularization

**Purpose:** Adds non-linear transformations to the data.

---

### 6. `train.py`
**Purpose:** Trains the model to learn patterns from your data.

**Training Process:**

1. **Setup** (Lines 7-12)
   - Moves model to GPU if available
   - Creates optimizer (AdamW) - adjusts model weights
   - Sets up TensorBoard for visualization

2. **Training Loop** (Lines 17-27)
   ```python
   for each epoch:
       for each batch:
           - Forward pass: Get predictions
           - Calculate loss: How wrong are predictions?
           - Backward pass: Calculate gradients
           - Update weights: Improve model
   ```

3. **Loss Function** (in model.py, line 35)
   - Cross-entropy loss
   - Measures difference between predicted and actual next character

4. **Checkpointing** (Line 29-30)
   - Saves model every 500 epochs
   - Allows resuming training or generating text

**Hyperparameters:**
- `num_epochs = 5000`: How many times to go through data
- `batch_size = 32`: Sequences per batch
- `lr = 1e-3`: Learning rate (how fast to learn)

**Monitoring:**
- Prints loss every 100 batches
- Logs to TensorBoard: `tensorboard --logdir=runs`

---

### 7. `generate.py`
**Purpose:** Uses trained model to generate new text.

**Generation Process:**

1. **Model Loading** (Lines 26-36)
   - Finds latest checkpoint (`model_epoch_*.pth`)
   - Loads trained weights into model

2. **Prompt Handling** (Lines 41-61)
   - Accepts user prompt (or starts from beginning)
   - Encodes prompt to token indices
   - Handles unknown characters gracefully

3. **Generation Function** (Lines 13-24)
   ```python
   for each new token to generate:
       - Get model prediction
       - Apply temperature scaling
       - Sample from probability distribution
       - Append to sequence
   ```

**Key Parameters:**
- `max_new_tokens = 500`: How many characters to generate
- `temperature = 0.8`: Controls randomness
  - Low (0.1): More deterministic, repetitive
  - High (2.0): More creative, random
  - Medium (0.8): Balanced

**Sampling:**
- Uses `torch.multinomial` to randomly sample based on probabilities
- Temperature scaling: `logits / temperature`
  - Higher temperature → flatter distribution → more random
  - Lower temperature → sharper distribution → more confident

---

## 🔑 Key Concepts Explained

### 1. **Embeddings**
- Convert discrete tokens (characters) into continuous vectors
- Similar characters get similar vectors (learned during training)
- Enables the model to understand relationships

### 2. **Self-Attention**
- Model learns to focus on relevant parts of the input
- Calculates "importance" of each character relative to others
- Enables understanding of context and relationships

### 3. **Causal Masking**
- Prevents model from "cheating" by looking at future tokens
- Only allows attention to previous tokens
- Essential for autoregressive generation

### 4. **Autoregressive Generation**
- Generates one token at a time
- Each new token depends on all previous tokens
- Process: `[start] → [start, token1] → [start, token1, token2] → ...`

### 5. **Temperature Sampling**
- Controls randomness in generation
- Scales logits before softmax
- Lower = more confident/predictable, Higher = more diverse

### 6. **Loss Function (Cross-Entropy)**
- Measures how well model predicts next character
- Lower loss = better predictions
- Used to update model weights via backpropagation

---

## 🔄 How Everything Works Together

### Complete Flow:

```
1. Raw Text (data_set.txt)
   ↓
2. prepare_data.py → Cleaned Text (cleaned_dataset.txt)
   ↓
3. tokenizer.py → Tokenized Data (numbers) + Vocabulary
   ↓
4. model.py → Neural Network Architecture
   ↓
5. train.py → Trained Model Weights (model_epoch_*.pth)
   ↓
6. generate.py → Generated Text Output
```

### Data Flow During Training:

```
Text → Characters → Indices → Embeddings → Transformer → Predictions → Loss → Backprop → Updated Weights
```

### Data Flow During Generation:

```
Prompt → Indices → Embeddings → Transformer → Probabilities → Sample → New Token → Repeat
```

---

## 📋 Step-by-Step Workflow

### Step 1: Setup Environment
```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Your Data
```bash
# Place your text in data_set.txt, then:
python prepare_data.py
```
**Output:** `cleaned_dataset.txt`

### Step 3: Train the Model
```bash
python train.py
```
**What happens:**
- Loads and tokenizes data
- Initializes model
- Trains for 5000 epochs
- Saves checkpoints every 500 epochs
- Logs loss to TensorBoard

**Monitor training:**
```bash
# In another terminal
tensorboard --logdir=runs
# Open http://localhost:6006 in browser
```

### Step 4: Generate Text
```bash
# With prompt
python generate.py "hello world"

# Without prompt (interactive)
python generate.py
```

**Output:** Generated text based on learned patterns

---

## 🎓 Learning Path

### Beginner Level:
1. Understand what each file does
2. Run the complete pipeline
3. Experiment with different prompts
4. Adjust `temperature` in `generate.py`

### Intermediate Level:
1. Modify hyperparameters in `model.py`:
   - `n_embd`: Embedding dimension
   - `n_head`: Number of attention heads
   - `n_layer`: Number of transformer blocks
2. Change training parameters in `train.py`:
   - Learning rate
   - Number of epochs
   - Batch size
3. Experiment with different data

### Advanced Level:
1. Implement different tokenization (word-level, subword)
2. Add more features (beam search, top-k sampling)
3. Implement fine-tuning
4. Optimize for your specific use case

---

## 🐛 Common Issues & Solutions

### Issue: "No model checkpoint found"
**Solution:** Train the model first using `train.py`

### Issue: "Character not in vocabulary"
**Solution:** The character wasn't in training data. Use characters that appeared in `data_set.txt`

### Issue: Model generates gibberish
**Solutions:**
- Train for more epochs
- Use more training data
- Adjust temperature (try lower values)
- Check if data was cleaned properly

### Issue: Training is slow
**Solutions:**
- Use GPU if available (CUDA)
- Reduce batch size
- Reduce `block_size` or `n_layer`
- Use smaller dataset

---

## 📊 Model Architecture Summary

```
Input: Character indices
  ↓
Token Embedding (vocab_size → n_embd)
  ↓
Position Embedding (position → n_embd)
  ↓
[Transformer Block × n_layer]
  ├─ Multi-Head Attention
  ├─ Feed-Forward Network
  └─ Layer Norm + Residuals
  ↓
Final Layer Norm
  ↓
Output Head (n_embd → vocab_size)
  ↓
Output: Probability distribution over vocabulary
```

---

## 🎯 Key Takeaways

1. **Character-level tokenization** makes this simple but limited
2. **Transformer architecture** enables understanding of context
3. **Self-attention** allows model to focus on relevant parts
4. **Autoregressive generation** creates text one token at a time
5. **Temperature sampling** controls creativity vs. accuracy trade-off

---

## 🚀 Next Steps

1. **Experiment with your own data**
2. **Tune hyperparameters** for better results
3. **Visualize attention** to see what model focuses on
4. **Try different architectures** (more layers, different sizes)
5. **Implement advanced sampling** (top-k, nucleus sampling)

---

## 📚 Additional Resources

- **Transformer Paper:** "Attention Is All You Need" (Vaswani et al., 2017)
- **GPT Paper:** "Improving Language Understanding by Generative Pre-Training"
- **PyTorch Tutorials:** Official PyTorch documentation
- **Andrej Karpathy's Blog:** Great explanations of transformers

---

**Happy Learning! 🎉**

This project is a great starting point for understanding how modern language models work. Experiment, break things, fix them, and most importantly - have fun learning!
