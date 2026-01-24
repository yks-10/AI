from torch.utils.data import Dataset, DataLoader
import torch

class TextDataset(Dataset):
    def __init__(self, text: str, block_size: int = 128):
        # Character-level tokenization
        self.chars = sorted(list(set(text)))  # Unique characters
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # Char to index
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # Index to char
        
        # Encode entire text
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        self.block_size = block_size  # Sequence length for training
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]  # Input + target (next token)
        return chunk[:-1], chunk[1:]  # x (input), y (target)

# Load and tokenize
with open('cleaned_dataset.txt', 'r') as f:
    text = f.read()

block_size = min(128, len(text) - 1) if len(text) > 1 else 1
dataset = TextDataset(text, block_size=block_size)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Vocab size: {dataset.vocab_size}")
print(f"Sample input: {''.join(dataset.itos[i.item()] for i in dataset[0][0][:20])}")