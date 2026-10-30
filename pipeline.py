import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from model import RPSClassifier
from evaluation import evaluate, confusion_matrix
from train import train

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30

# Set device device based on whats available
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Image to tensor
to_tensor = v2.Compose([
    v2.ToImage(),
    v2.Resize((128,128)),
    v2.ToDtype(torch.float32, scale=True)
])

# Load datasets and convert to tensors
train_dataset = ImageFolder("LFD2000-resources/datasets/RPS/train", transform=to_tensor)
test_dataset = ImageFolder("LFD2000-resources/datasets/RPS/test", transform=to_tensor)
class_names = train_dataset.classes

# Load data with a 12/87 test/training split
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False)

sample_image, _ = train_dataset[0]
input_size = sample_image.numel()

def train_and_evaluate(num_epochs):
    torch.manual_seed(52)
    model = RPSClassifier(num_classes=len(class_names)).to(device)

    train(model, LEARNING_RATE, num_epochs, train_loader, device)
    predictions, labels = evaluate(model, test_loader, device)

    confusion_matrix(predictions, labels, class_names)

train_and_evaluate(NUM_EPOCHS)