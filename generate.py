import torch
from torch.nn import functional as F
from tokenizer import dataset
from model import SimpleGPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleGPT(vocab_size=dataset.vocab_size)
model.to(device)

def generate(model, idx, max_new_tokens: int = 100, temperature: float = 0.8):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
            logits = model(idx_cond)
            logits = logits[0, -1, :] / temperature  # Last token, scale
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx

# Load trained model
model.load_state_dict(torch.load('model_epoch_4500.pth'))  # Last checkpoint

# Generate
context = torch.tensor([[0]], device=device)  # Start with first character in vocab
generated = generate(model, context, max_new_tokens=200)
print(''.join(dataset.itos[i.item()] for i in generated[0]))