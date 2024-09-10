import torch.nn as nn 
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, input_size, ouput_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.ln1 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 32)
        self.ln3 = nn.LayerNorm(32)
        self.fc4 = nn.Linear(32, 4)
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.fc4(x)


