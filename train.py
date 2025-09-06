import random
import itertools as it
import os
import pickle
import time
import tqdm

import pandas as pd
import torch
import torch.nn.functional as F
import h5py

from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import MultiStepLR
from dotenv import load_dotenv
from torch.utils.data import DataLoader

import data_preprocess_mol as preprocess_mol
import layers as layers
import networks as network


class ModelResult(object):

    def __init__(self, batch_size, learning_rate, n_type, n_head, e_size, n_conv):

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.n_type = n_type
        self.n_head = n_head
        self.e_size = e_size
        self.n_conv = n_conv

        self.t_losses = []
        self.v_losses = []


def pp_loss(values: list):
    n = len(values)
    mean = sum(values) / n
    sd = sum(map(lambda x: (x - mean)**2, values)) / n
    return mean, sd, f"{mean:2.5f}±{sd:2.2f}"


def train_model(model, num_epochs, trainloader, criterion, optimizer,
                scheduler, validloader, verbose=False):
    """
    Training loop for the torch model
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_running_loss = []
        valid_loss = []
        # train_acc = 0
        # valid_acc = 0

        # Training phase
        model.train()
        start = time.time()

        # l_tl = len(trainloader)
        for graphs, labels in tqdm.tqdm(trainloader, desc=f"Epoch ({epoch:2.0f})"):

            # print(graphs, graphs.adj_mat.shape, graphs.emb_mat.shape)

            graphs = graphs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            value = model(graphs)

            # print(value, labels, graphs)

            # print(f"Batch {i:4.0f}/{l_tl:4.0f} accuracy: {get_accuracy(value, labels)}" )

            loss = criterion(value, labels)
            loss.backward()
            optimizer.step()

            train_running_loss.append(loss.item())
            # train_acc += get_accuracy(value, labels)

        # Validation phase
        model.eval()
        with torch.no_grad():
            for graph, labels in tqdm.tqdm(validloader, desc="Validation"):
                graphs = graph.to(device)
                labels = labels.to(device)
                value = model(graphs)
                loss = criterion(value, labels)
                valid_loss.append(loss.item())
                # valid_acc += get_accuracy(value, labels)

        # Scheduler step
        # scheduler.step(valid_loss)
        scheduler.step()

        # Calculate average losses
        avg_train_loss = pp_loss(train_running_loss)[0]
        avg_valid_loss = pp_loss(valid_loss)[0]
        # avg_train_acc = train_acc / len(trainloader)
        # avg_valid_acc = valid_acc / len(validloader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_valid_loss)

        with open("./model-outputs/model_results.pkl", "wb") as f:
            print("Saving...")
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

        if verbose:
            lr = scheduler.get_last_lr()[0]
            print(
                f"\t(sd) T_Loss: {pp_loss(train_running_loss)[-1]} | (sd) V_Loss: {pp_loss(valid_loss)[-1]}"
                # + f" | T_acc: {avg_train_acc:2.2f} " + "| V_acc: {avg_valid_acc:2.2f}"
                + f" | Time: {time.time()-start:2.2f} | lr: {lr:2.5f}"
            )

    return train_losses, val_losses


def KF_train_valid(
    t_data,
    n_type,
    n_head,
    e_size,
    n_conv,
    X_col,
    y_col,
    n_fold=5,
    batch_size=8,
    learning_rate=0.01,
    milestones=[5, 15],
    device=None,
):

    results = {}
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    fold_indices = list(kf.split(t_data))

    perameter_combinations = list(it.product(n_type, n_head, e_size, n_conv))
    count = 0
    num_models = len(perameter_combinations)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for p in perameter_combinations:

        n_type, n_head, e_size, n_conv = p
        count += 1

        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            mr = ModelResult(*p)

            print(f"Model {count}/{num_models}, Fold {fold_idx + 1}/{n_fold}")

            # Creating model
            model = network.GAT(
                emb_dim=11, out_dim=1, embedding_size=e_size, n_conv=n_conv, act=F.gelu
            ).to(device)

            # Initializing training parameters
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = MultiStepLR(
                optimizer, milestones=milestones, gamma=0.1)
            criterion = torch.nn.CrossEntropyLoss()

            # Setting training and validation data
            train_data = t_data.iloc[train_idx].reset_index(drop=True)
            val_data = t_data.iloc[val_idx].reset_index(drop=True)

            trainloader = torch.utils.data.DataLoader(
                preprocess.PandasDataset(train_data, X_col, y_col),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=preprocess.graph_collate_fn,
            )

            validloader = torch.utils.data.DataLoader(
                preprocess.PandasDataset(val_data, X_col, y_col),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=preprocess.graph_collate_fn,
            )

            n_epochs = sum(milestones)

            # Training
            t_loss, v_loss = train_model(
                model,
                n_epochs,
                trainloader,
                criterion,
                optimizer,
                scheduler,
                validloader,
                verbose=True,
            )

            # Saving and outputting intermediate results
            mr.t_losses.append(t_loss)
            mr.v_losses.append(v_loss)

            print("T:", t_loss)
            print("V:", v_loss)

            results[(*p, fold_idx)] = mr

        # Dumping onto a file to save
        with open("./model-outputs/model_results.pkl", "wb") as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

###############################################################################


def main():

    load_dotenv()

    # Set arguments
    mol_data_path = "./mol-data/mol_sample_metadata.csv"
    metadata = pd.read_csv(mol_data_path)

    # Path to the h5py file containing all the preprocessed embeddings, labels
    # and adjacency matrix
    treated_molecules_path = os.getenv("TREATED_MOLECULES_PATH")
    if treated_molecules_path is None:
        raise ValueError("Invalid .h5 data path.")

    h5f = h5py.File(treated_molecules_path, "r")

    # Training parameters
    learning_rate = 0.1
    milestones = [400]  # [5, 80, 80]
    n_epochs = sum(milestones)
    batch_size = 8
    num_workers = 4
    prefetch_factor = 8
    step_size = 5


    print(metadata)

    tset = metadata[metadata["set"] == "train"]
    md1 = tset.iloc[:int(tset.shape[0] * 0.2), :].reset_index(drop=True)
    md2 = tset.iloc[int(tset.shape[0] * 0.2):, :].reset_index(drop=True)

    datasett = preprocess_mol.H5PyMolDataset(h5f, md2, "idx", train=True)
    dataloadert = DataLoader(
        datasett, batch_size=batch_size,
        shuffle=False, collate_fn=preprocess_mol.graph_collate_fn_h5)

    datasetv = preprocess_mol.H5PyMolDataset(h5f, md1, "idx", train=True)
    dataloaderv = DataLoader(
        datasetv, batch_size=batch_size,
        shuffle=False, collate_fn=preprocess_mol.graph_collate_fn_h5)

    print(f"num_workers={num_workers}, prefetch_factor={prefetch_factor}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Runnung one device {device}")

    model = network.GraphGCNN(
        in_dim=19, out_dim=1, upscale_dim=128, act=F.gelu)

    # model = network.MultipleGAT(
    #     emb_dim=19, out_dim=1, upscale_size=19, act=F.gelu)

    model = model.to(device)
    print(f"Model: \n{model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Optimizer:{optimizer}")

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, patience=5, factor=2e-1)

    # scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=0.80,
    )

    criterion = torch.nn.MSELoss()

    t_loss, v_loss = train_model(
        model,
        n_epochs,
        dataloadert,
        criterion,
        optimizer,
        scheduler,
        dataloaderv,
        verbose=True,
    )

    print(t_loss, v_loss)


if __name__ == "__main__":
    main()
