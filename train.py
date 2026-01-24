import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter  # Optional: pip install tensorboard
from tokenizer import dataset, dataloader
from model import SimpleGPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleGPT(vocab_size=dataset.vocab_size)
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-3)
writer = SummaryWriter('runs/textgen')  # For logging

num_epochs = 5000  # Adjust based on dataset size
batch_size = 32

for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)
    
    if epoch % 500 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')

print("Training complete! Saved model.")