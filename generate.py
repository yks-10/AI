import torch
from torch.nn import functional as F
import glob
import os
import sys
from tokenizer import dataset
from model import SimpleGPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleGPT(vocab_size=dataset.vocab_size)
model.to(device)

def generate(model, idx, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50):
    """
    Generate text using the model with top-k sampling to reduce repetition.
    
    Args:
        model: The trained model
        idx: Input token indices
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Only sample from top-k most likely tokens (0 = disabled)
    """
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
            logits = model(idx_cond)
            logits = logits[0, -1, :] / temperature  # Last token, scale
            
            # Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx

# Find best model first, then fall back to latest checkpoint
model_to_load = None
if os.path.exists('model_best.pth'):
    model_to_load = 'model_best.pth'
    print("Found best model checkpoint, loading it...")
else:
    # Find latest model checkpoint
    model_files = glob.glob('model_epoch_*.pth')
    if not model_files:
        raise FileNotFoundError("No model checkpoint found! Please train the model first.")
    
    # Get the latest checkpoint by epoch number
    model_to_load = max(model_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
    print(f"Loading latest model: {model_to_load}")

# Load trained model
model.load_state_dict(torch.load(model_to_load, map_location=device))
print("Model loaded successfully!")

# Generate text
# You can provide a custom prompt via command line argument, or start from the beginning
if len(sys.argv) > 1:
    prompt = sys.argv[1]
    print(f"Using prompt: '{prompt}'")
else:
    try:
        prompt = input("Enter a prompt (or press Enter to start from beginning): ").strip()
    except EOFError:
        # Non-interactive mode, start from beginning
        prompt = ""
        print("No prompt provided, starting from beginning...")

if prompt:
    # Encode the prompt using BPE tokenizer
    try:
        token_ids = dataset.tokenizer.encode(prompt)
        context = torch.tensor([token_ids], device=device)
        print(f"Encoded prompt to {len(token_ids)} tokens")
    except Exception as e:
        print(f"Warning: Error encoding prompt: {e}. Starting from beginning.")
        context = torch.tensor([[0]], device=device)
else:
    # Start with first token in vocab
    context = torch.tensor([[0]], device=device)

# Generate with improved sampling
print("\nGenerating text...")
# Use top-k sampling to reduce repetition (top_k=50 means only sample from top 50 most likely tokens)
generated = generate(model, context, max_new_tokens=500, temperature=1.0, top_k=50)
# Decode using BPE tokenizer
token_ids = generated[0].cpu().tolist()
output = dataset.tokenizer.decode(token_ids)
print("\n" + "="*50)
print("Generated text:")
print("="*50)
print(output)
print("="*50)