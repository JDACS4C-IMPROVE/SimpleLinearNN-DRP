import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

def determine_input_dim(dataloader):
    sample_data = next(iter(dataloader)) # Get first batch
    print("sample_data[0]:", sample_data[0])
    print("sample_data[0].shape[1]:", sample_data[0].shape[1])
    input_dim = sample_data[0].shape[1]  # Extract num_genes
    return input_dim

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, dropout_prob):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear1 = nn.Linear(input_dim, np.floor(input_dim/2).astype(int))
        self.linear2 = nn.Linear(np.floor(input_dim/2).astype(int), np.floor(input_dim/4).astype(int))
        self.linear3 = nn.Linear(np.floor(input_dim/4).astype(int), 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

class DRPDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

def predicting(model, data_loader, device):
    # Inside your training loop, after an epoch:
    model.eval() # Set model to evaluation mode (disables dropout, batch norm updates)
    all_labels = []
    all_predictions = []

    with torch.no_grad(): # Disable gradient calculation during validation
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            #print("pred outputs:", outputs.cpu().numpy().flatten())
            all_labels.extend(labels.tolist())
            all_predictions.extend(outputs.cpu().numpy().flatten())

    model.train() # Set model back to training mode
    return all_labels, all_predictions
