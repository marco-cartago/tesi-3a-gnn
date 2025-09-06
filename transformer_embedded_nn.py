import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

# Load data
mol_data = pd.read_csv("./molecs/MyRoboBohr2.csv")
np_data = np.load("./molecs/SMILES.npz")


class Data(Dataset):

    def __init__(self):
        self.mol_data = mol_data
        self.np_data = np_data

    def __getitem__(self, idx):
        s = self.mol_data["Canonical SMILES"][idx]
        data = self.np_data[s].astype(np.float32)
        e = torch.from_numpy(data).mean(dim=0)
        l = torch.tensor(self.mol_data["Eat"][idx], dtype=torch.float32)

        return (e, l)

    def __len__(self):
        return len(self.mol_data)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def main():

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset = Data()
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(10):
        for i, (e, l) in enumerate(data_loader):
            e = e.to(device)
            l = l.to(device)
            outputs = model(e)

            loss = criterion(outputs, l.view(-1, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch {}: Loss = {:.4f}'.format(epoch+1, loss.item()))


if __name__ == "__main__":
    main()
