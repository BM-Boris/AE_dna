import torch
import torch.nn as nn
import os

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv1d(1, 9, 9, stride=1, padding = 1),  
            nn.BatchNorm1d(9),  
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Conv1d(9, 18, 9, stride=1, padding = 1),
            nn.BatchNorm1d(18), 
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Conv1d(18, 27, 9, stride=1, padding = 1),
            nn.BatchNorm1d(27), 
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(8559, 780),  
            nn.ReLU(),
            nn.Linear(780, 128),  
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        # Flatten the tensor
        x = self.classifier(x)
        return x