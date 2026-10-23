import torch

def evaluate(model, test_loader, device):
    """Run testing predictions"""
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

        print(f"Test accuracy: {accuracy:.2%}")
        return predictions
    
    except Exception as e:
        return f"Error in evaluating accurary: {e}"

def confusion_matrix(predictions, test_loader, class_names):
    """Evaluate confusion matrix, precision and recall from evaluation"""
    try:
        num_classes = len(class_names)
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

        for _, labels in test_loader:
            for true_label, predicted_label in zip(labels, predictions):
                confusion[true_label, predicted_label] += 1

        print("Confusion matrix:")
        print(confusion)

        correct_per_class = confusion.diag()
        precision = correct_per_class / confusion.sum(dim=0)
        recall = correct_per_class / confusion.sum(dim=1)

        for i, name in enumerate(class_names):
            print(f"{name:>10}: precision = {precision[i]:.2f}, recall = {recall[i]:.2f}")
    except Exception as e:
        return f"Error in calculating confusion matrix: {e}"