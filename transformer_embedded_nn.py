from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim


class Data(Dataset):

    def __init__(self, mol_data, np_data):
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

    # Load data
    mol_data = pd.read_csv("./mol-data/MyRoboBohr2.csv")

    # Was too large to be loaded on github.
    np_data = np.load("./mol-data/transformer_embedded.npz")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    full_dataset = Data(mol_data, np_data)

    val_frac = 0.2
    val_size = int(len(full_dataset) * val_frac)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 200

    for epoch in range(num_epochs):

        # Training
        model.train()
        train_loss = 0.0

        for _, (e, l) in enumerate(train_loader):
            e = e.to(device)
            l = l.to(device)
            outputs = model(e)

            loss = criterion(outputs, l.view(-1, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * e.size(0)

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for e_val, l_val in val_loader:
                e_val = e_val.to(device)
                l_val = l_val.to(device)
                outputs_val = model(e_val)
                loss_val = criterion(outputs_val, l_val.view(-1, 1))
                val_loss += loss_val.item() * e_val.size(0)

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss = {val_loss:.4f}")


if __name__ == "__main__":
    main()
