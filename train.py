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

num_epochs = 50  # Adjust based on dataset size
batch_size = 32

# Track best loss for saving best model
best_loss = float('inf')

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)
    
    # Calculate average loss for the epoch
    avg_loss = epoch_loss / num_batches if num_batches > 0 else epoch_loss
    print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
    writer.add_scalar('Loss/epoch', avg_loss, epoch)
    
    # Save checkpoint every 10 epochs or if this is the best model so far
    if epoch % 10 == 0 or avg_loss < best_loss:
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
        print(f"Saved checkpoint: model_epoch_{epoch}.pth")
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Also save as best model
            torch.save(model.state_dict(), 'model_best.pth')
            print(f"New best model saved! Loss: {avg_loss:.4f}")

# Always save final model
torch.save(model.state_dict(), f'model_epoch_{num_epochs-1}.pth')
print(f"\nTraining complete! Final model saved: model_epoch_{num_epochs-1}.pth")
print(f"Best model saved: model_best.pth (Loss: {best_loss:.4f})")