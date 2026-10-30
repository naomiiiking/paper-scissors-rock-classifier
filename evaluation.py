import torch

def evaluate(model, test_loader, device):
    """Run testing predictions"""
    model.eval()
    try:
        all_preds, all_labels = [], []
        with torch.inference_mode():
            for images, labels in test_loader:
                logits = model(images.to(device))
                all_preds.append(logits.argmax(dim=1).cpu())
                all_labels.append(labels)
        predictions = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        accuracy = (predictions == labels).float().mean().item()

        print(f"Test accuracy: {accuracy:.2%}")
        return predictions, labels
        
    except Exception as e:
        return f"Error in evaluating accurary: {e}"

def confusion_matrix(predictions, labels, class_names):
    """Evaluate confusion matrix, precision and recall from evaluation"""
    try:
        num_classes = len(class_names)
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

        for t, p in zip(labels, predictions):
            confusion[t, p] += 1

        print("Confusion matrix:")
        print(confusion)

        correct_per_class = confusion.diag()
        precision = correct_per_class / confusion.sum(dim=0)
        recall = correct_per_class / confusion.sum(dim=1)

        for i, name in enumerate(class_names):
            print(f"{name:>10}: precision = {precision[i]:.2f}, recall = {recall[i]:.2f}")
    except Exception as e:
        return f"Error in calculating confusion matrix: {e}"