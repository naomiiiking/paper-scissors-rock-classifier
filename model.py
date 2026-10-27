from torch import nn

class RPSClassifier(nn.Module):
    """Model class with construction and forward function. Flattens vector, creates first layer, applies activation, returns logits"""
    def __init__(self, input_size, hidden_size=128, num_classes=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.01)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
        