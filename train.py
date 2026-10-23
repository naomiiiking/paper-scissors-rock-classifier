import torch
from torch import nn

def train(model, LEARNING_RATE, NUM_EPOCHS, train_loader, device):
    try:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)          # 1. Forward pass
                loss = loss_fn(logits, labels)  # 2. Loss computation
                optimizer.zero_grad()           # 3. Clear old gradients
                loss.backward()                 # 4. Backward pass
                optimizer.step()                 # 5. Param update

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch}: loss = {avg_loss:.4f}")
    except Exception as e:
        raise e

