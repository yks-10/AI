import torch
import torch.nn as nn
from torch.nn import functional as F

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int = 128, n_head: int = 4, n_layer: int = 4, block_size: int = 128):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer norm
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        
        # Output head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, idx, targets=None):
        b, t = idx.shape
        tok_emb = self.token_embedding(idx)  # (b, t, n_embd)
        pos_emb = self.position_embedding(torch.arange(t, device=idx.device))  # (t, n_embd)
        x = tok_emb + pos_emb  # Add positional encoding
        x = self.blocks(x)  # Apply transformer blocks
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (b, t, vocab_size)
        
        if targets is None:
            return logits
        else:
            # Cross-entropy loss (shifted for next-token prediction)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd)  # Self-attention
        self.ffwd = FeedForward(n_embd)  # Feed-forward
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Residual + attention
        x = x + self.ffwd(self.ln2(x))  # Residual + FFN
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, n_embd * 3)  # Q, K, V
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        b, t, c = x.shape  # Batch, time, channels
        qkv = self.c_attn(x).reshape(b, t, 3, self.n_head, self.head_size).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, nh, t, hs)
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))  # Causal mask not shown for brevity
        att = att.masked_fill(torch.tril(torch.ones(t, t, device=x.device)) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v  # (b, nh, t, hs)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.c_proj(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.net(x)

# Model is instantiated in train.py with the correct vocab_size