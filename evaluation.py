import torch

def evaluate(model, test_loader, device):
    try:
        model.eval()
        correct = 0
        total = 0

        with torch.inference_mode():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                predictions = logits.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        return f"Test accuracy: {accuracy:.2%}"
    except Exception as e:
        return e