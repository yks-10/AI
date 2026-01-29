from torch.utils.data import Dataset, DataLoader
import torch
import re
from collections import defaultdict
import json
import os

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer
    
    BPE is a subword tokenization algorithm that:
    1. Starts with character-level vocabulary
    2. Iteratively merges most frequent pairs of tokens
    3. Creates a vocabulary of subword units
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word_freqs = defaultdict(int)
        self.splits = {}  # Word -> list of characters
        self.merges = {}  # (token1, token2) -> merged_token
        self.vocab = {}  # token -> index
        self.inverse_vocab = {}  # index -> token
        
    def _get_word_freqs(self, text: str):
        """Extract words and their frequencies from text"""
        # Split text into words (keeping spaces as separate tokens)
        words = re.findall(r'\S+|\s+', text)
        self.word_freqs = defaultdict(int)
        for word in words:
            self.word_freqs[word] += 1
        return self.word_freqs
    
    def _get_splits(self):
        """Initialize splits: each word as list of characters"""
        self.splits = {}
        for word in self.word_freqs:
            # Split word into characters, add end-of-word token
            self.splits[word] = list(word) + ['</w>']
        return self.splits
    
    def _get_stats(self):
        """Count frequency of adjacent token pairs"""
        pairs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            for i in range(len(split) - 1):
                pairs[(split[i], split[i + 1])] += freq
        return pairs
    
    def _merge_pair(self, pair: tuple):
        """Merge the most frequent pair in all splits"""
        a, b = pair
        new_splits = {}
        bigram = ''.join(pair)
        
        for word in self.splits:
            split = self.splits[word]
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                    new_split.append(bigram)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        
        self.splits = new_splits
        self.merges[pair] = bigram
    
    def train(self, text: str):
        """Train BPE tokenizer on text"""
        print("Training BPE tokenizer...")
        
        # Get word frequencies
        self._get_word_freqs(text)
        print(f"Found {len(self.word_freqs)} unique words")
        
        # Initialize splits (character-level)
        self._get_splits()
        
        # Get initial vocabulary (all unique characters + </w>)
        chars = set()
        for word in self.word_freqs:
            for char in word:
                chars.add(char)
        chars.add('</w>')
        
        # Initialize vocabulary
        vocab = sorted(list(chars))
        num_merges = self.vocab_size - len(vocab)
        
        if num_merges <= 0:
            print(f"Warning: vocab_size ({self.vocab_size}) <= initial vocab size ({len(vocab)})")
            num_merges = 100  # Default to 100 merges
        
        print(f"Starting with {len(vocab)} base tokens, will perform {num_merges} merges")
        
        # Perform merges
        for i in range(num_merges):
            pairs = self._get_stats()
            if not pairs:
                break
            
            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)
            self._merge_pair(best_pair)
            
            # Add merged token to vocabulary
            merged_token = self.merges[best_pair]
            vocab.append(merged_token)
            
            if (i + 1) % 100 == 0:
                print(f"Merge {i + 1}/{num_merges}: merged '{best_pair[0]}' + '{best_pair[1]}' -> '{merged_token}' (freq: {pairs[best_pair]})")
        
        # Build final vocabulary
        self.vocab = {token: idx for idx, token in enumerate(vocab)}
        self.inverse_vocab = {idx: token for idx, token in enumerate(vocab)}
        self.vocab_size = len(self.vocab)
        
        print(f"BPE training complete! Final vocab size: {self.vocab_size}")
        return self
    
    def encode(self, text: str) -> list:
        """Encode text into token indices"""
        # Split text into words
        words = re.findall(r'\S+|\s+', text)
        tokens = []
        
        for word in words:
            # Start with character-level split
            split = list(word) + ['</w>']
            
            # Apply all learned merges
            while len(split) > 1:
                pairs = [(split[i], split[i + 1]) for i in range(len(split) - 1)]
                
                # Find the pair that exists in merges (prioritize longest matches)
                pair_to_merge = None
                for pair in pairs:
                    if pair in self.merges:
                        pair_to_merge = pair
                        break
                
                if pair_to_merge is None:
                    break
                
                # Merge the pair
                a, b = pair_to_merge
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                        new_split.append(self.merges[pair_to_merge])
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                split = new_split
            
            # Convert to indices
            for token in split:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    # Fallback: use unknown token or first token
                    tokens.append(0)
        
        return tokens
    
    def decode(self, token_ids: list) -> str:
        """Decode token indices back to text"""
        tokens = [self.inverse_vocab.get(idx, '') for idx in token_ids]
        text = ''.join(tokens)
        # Remove end-of-word markers and restore spaces
        text = text.replace('</w>', ' ')
        return text
    
    def save(self, filepath: str):
        """Save tokenizer to file"""
        data = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'inverse_vocab': self.inverse_vocab,
            'merges': {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load tokenizer from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.vocab = {k: int(v) for k, v in data['vocab'].items()}
        self.inverse_vocab = {int(k): v for k, v in data['inverse_vocab'].items()}
        self.merges = {tuple(k.split('|')): v for k, v in data['merges'].items()}
        print(f"Tokenizer loaded from {filepath}")
        return self


class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer: BPETokenizer, block_size: int = 128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Encode entire text
        token_ids = tokenizer.encode(text)
        self.data = torch.tensor(token_ids, dtype=torch.long)
        
        # Vocabulary mappings (for compatibility)
        self.vocab_size = tokenizer.vocab_size
        self.stoi = tokenizer.vocab
        self.itos = tokenizer.inverse_vocab
    
    def __len__(self):
        return max(0, len(self.data) - self.block_size)
    
    def __getitem__(self, idx):
        if idx + self.block_size + 1 > len(self.data):
            # Handle edge case
            idx = max(0, len(self.data) - self.block_size - 1)
        chunk = self.data[idx:idx + self.block_size + 1]  # Input + target (next token)
        return chunk[:-1], chunk[1:]  # x (input), y (target)


# Load and tokenize
print("Loading cleaned dataset...")
with open('cleaned_dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Initialize and train BPE tokenizer
tokenizer_path = 'bpe_tokenizer.json'
if os.path.exists(tokenizer_path):
    print(f"Loading existing tokenizer from {tokenizer_path}")
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
else:
    print("Training new BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=1000)  # Adjust vocab size as needed
    tokenizer.train(text)
    tokenizer.save(tokenizer_path)

# Create dataset
block_size = min(128, len(text) - 1) if len(text) > 1 else 1
dataset = TextDataset(text, tokenizer, block_size=block_size)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"\nDataset Info:")
print(f"Vocab size: {dataset.vocab_size}")
print(f"Total tokens: {len(dataset.data)}")
print(f"Sample tokens: {dataset.data[:20].tolist()}")
print(f"Sample decoded: {tokenizer.decode(dataset.data[:20].tolist())}")