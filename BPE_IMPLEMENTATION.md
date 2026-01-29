# Byte Pair Encoding (BPE) Implementation Guide

## What Changed?

The project has been updated from **character-level tokenization** to **Byte Pair Encoding (BPE)**, a subword tokenization method used by modern language models like GPT, BERT, and others.

## What is BPE?

**Byte Pair Encoding (BPE)** is a data compression algorithm adapted for tokenization:

1. **Starts** with character-level vocabulary (each character is a token)
2. **Iteratively merges** the most frequent pair of adjacent tokens
3. **Creates** a vocabulary of subword units (characters, character pairs, longer sequences)
4. **Result**: A vocabulary that balances between character and word-level approaches

### Example:
```
Initial: ["h", "e", "l", "l", "o", "</w>"]
After merge "l" + "l": ["h", "e", "ll", "o", "</w>"]
After merge "he" + "ll": ["hell", "o", "</w>"]
```

## Key Benefits

1. **Efficiency**: Fewer tokens than character-level → faster training/inference
2. **Flexibility**: Handles out-of-vocabulary words by breaking them into subwords
3. **Balance**: Better than pure character-level (too granular) or word-level (too rigid)
4. **Industry Standard**: Used by GPT-2, GPT-3, BERT, and other modern models

## Implementation Details

### BPETokenizer Class (`tokenizer.py`)

#### Key Methods:

1. **`train(text: str)`**
   - Trains BPE tokenizer on your text
   - Starts with character-level vocabulary
   - Performs merges until desired vocab size
   - Saves tokenizer to `bpe_tokenizer.json`

2. **`encode(text: str) -> list`**
   - Converts text to token indices
   - Splits text into words
   - Applies learned merges
   - Returns list of token IDs

3. **`decode(token_ids: list) -> str`**
   - Converts token indices back to text
   - Reconstructs original text from tokens

4. **`save(filepath: str)` / `load(filepath: str)`**
   - Persists tokenizer for reuse
   - Avoids retraining on every run

### How It Works

#### Training Phase:
```python
# 1. Count word frequencies
word_freqs = {"hello": 5, "world": 3, ...}

# 2. Initialize character-level splits
splits = {"hello": ["h", "e", "l", "l", "o", "</w>"], ...}

# 3. Find most frequent pair
pairs = {("l", "l"): 10, ("e", "l"): 8, ...}
best_pair = ("l", "l")

# 4. Merge the pair everywhere
splits = {"hello": ["h", "e", "ll", "o", "</w>"], ...}

# 5. Repeat until vocab size reached
```

#### Encoding Phase:
```python
text = "hello world"
# 1. Split into words: ["hello", " ", "world"]
# 2. Apply merges to each word
# 3. Convert to token IDs: [42, 15, 89, ...]
```

#### Decoding Phase:
```python
token_ids = [42, 15, 89, ...]
# 1. Convert IDs to tokens: ["hell", "o", " ", "wor", "ld"]
# 2. Join tokens: "hello world"
```

## Configuration

### Vocabulary Size
Default: **1000 tokens** (configurable in `tokenizer.py`)

```python
tokenizer = BPETokenizer(vocab_size=2000)  # Increase for larger datasets
```

**Recommendations:**
- Small datasets (< 1MB): 500-1000 tokens
- Medium datasets (1-10MB): 1000-2000 tokens
- Large datasets (> 10MB): 2000-5000 tokens

### Tokenizer Persistence
- First run: Trains and saves to `bpe_tokenizer.json`
- Subsequent runs: Loads existing tokenizer
- To retrain: Delete `bpe_tokenizer.json`

## File Changes

### `tokenizer.py`
- ✅ Added `BPETokenizer` class with full BPE implementation
- ✅ Updated `TextDataset` to use BPE tokenizer
- ✅ Maintains backward compatibility (same interface)

### `generate.py`
- ✅ Updated to use `tokenizer.encode()` instead of character-level encoding
- ✅ Updated to use `tokenizer.decode()` for output

### `train.py`
- ✅ No changes needed (uses dataset interface)

### `model.py`
- ✅ No changes needed (works with any vocab size)

## Usage

### First Run (Trains Tokenizer):
```bash
python prepare_data.py  # Clean data
python train.py         # Trains BPE + model
```

**Output:**
```
Training BPE tokenizer...
Found 1234 unique words
Starting with 27 base tokens, will perform 973 merges
Merge 100/973: merged 'e' + ' '</w> -> 'e </w>' (freq: 456)
...
BPE training complete! Final vocab size: 1000
Tokenizer saved to bpe_tokenizer.json
```

### Subsequent Runs:
```bash
python train.py  # Loads existing tokenizer
```

**Output:**
```
Loading existing tokenizer from bpe_tokenizer.json
Tokenizer loaded from bpe_tokenizer.json
```

## Comparison: Character vs BPE

### Character-Level (Old):
```
Text: "hello world"
Tokens: [h, e, l, l, o,  , w, o, r, l, d]  # 11 tokens
Vocab size: ~50-100 (all unique characters)
```

### BPE (New):
```
Text: "hello world"
Tokens: [hell, o,  , wor, ld]  # 5 tokens
Vocab size: 1000 (subword units)
```

**Benefits:**
- ✅ 50% fewer tokens → faster processing
- ✅ Better semantic understanding (common words as single tokens)
- ✅ Handles new words gracefully (breaks into known subwords)

## Troubleshooting

### Issue: "Tokenizer training is slow"
**Solution:** Reduce vocab size or use smaller dataset sample for training

### Issue: "Out of memory during training"
**Solution:** 
- Reduce vocab size
- Process text in chunks
- Use smaller batch size

### Issue: "Generated text has weird tokens"
**Solution:**
- Check if tokenizer was trained on similar data
- Retrain tokenizer with more data
- Increase vocab size

### Issue: "Want to retrain tokenizer"
**Solution:**
```bash
rm bpe_tokenizer.json
python train.py  # Will retrain
```

## Advanced Customization

### Custom Vocabulary Size:
```python
# In tokenizer.py, line ~200
tokenizer = BPETokenizer(vocab_size=2000)  # Change this
```

### Custom Merge Count:
```python
# In BPETokenizer.train(), adjust num_merges
num_merges = self.vocab_size - len(vocab)
# Or set directly:
num_merges = 500  # Fixed number of merges
```

### Add Special Tokens:
```python
# In BPETokenizer.__init__()
self.special_tokens = {'<unk>': 0, '<pad>': 1, '<start>': 2, '<end>': 3}
# Add to vocab initialization
```

## Performance Impact

### Training Speed:
- **Faster**: Fewer tokens per sequence → less computation
- **Memory**: Similar (vocab size similar to character-level)

### Generation Quality:
- **Better**: Model learns meaningful subword units
- **More coherent**: Common words as single tokens

### Model Size:
- **Similar**: Embedding size depends on vocab size
- **Efficient**: 1000-token vocab is manageable

## Next Steps

1. **Experiment with vocab size**: Try 500, 1500, 2000
2. **Compare results**: Character-level vs BPE
3. **Add special tokens**: `<unk>`, `<pad>`, etc.
4. **Implement other tokenizers**: SentencePiece, WordPiece

## References

- **Original BPE Paper**: "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)
- **GPT-2 Tokenizer**: Uses BPE with byte-level encoding
- **HuggingFace Tokenizers**: Modern BPE implementation

---

**The BPE implementation is complete and ready to use!** 🎉
