import os
import time
import pickle
import tqdm

from dotenv import load_dotenv
from sklearn.model_selection import KFold

import torch
import pandas as pd
import h5py

import networks as nets

MODEL_CLASSES = [
    cls for name, cls in vars(nets).items()
    if isinstance(cls, type) and ("GAT" in name or "GCNN" in name)
]

IN_EMB_DIM = 19
OUT_EMB_DIM = 1
UPSCALE_DIM = 128

MODELS = {

    # To test a model that just does aggregation
    "JustAggrConvGCNN": nets.JustAggrConvGCNN(
        in_dim=IN_EMB_DIM, out_dim=OUT_EMB_DIM, upscale_dim=UPSCALE_DIM),

    # A single convolution with average aggregation
    "OneConvGCNN": nets.OneConvGCNN(
        in_dim=IN_EMB_DIM, out_dim=OUT_EMB_DIM, upscale_dim=UPSCALE_DIM),

    # Two convolutions with average aggregation
    "TwoConvGCNN": nets.TwoConvGCNN(
        in_dim=IN_EMB_DIM, out_dim=OUT_EMB_DIM, upscale_dim=UPSCALE_DIM),

    # Graph attention network with one head
    "OneHeadGAT": nets.OneHeadGAT(
        in_dim=IN_EMB_DIM, out_dim=OUT_EMB_DIM, upscale_dim=UPSCALE_DIM),

    # Graph attention network with two heads
    "MultiHeadGAT-2": nets.MultiHeadGAT(
        in_dim=IN_EMB_DIM, out_dim=OUT_EMB_DIM, upscale_dim=UPSCALE_DIM,
        n_head=2),

    # Graph attention with four heads
    "MultiHeadGAT-4": nets.MultiHeadGAT(
        in_dim=IN_EMB_DIM, out_dim=OUT_EMB_DIM, upscale_dim=UPSCALE_DIM,
        n_head=4),
}


def ploss(values: list):
    n = len(values)
    mean = sum(values) / n
    sd = sum(map(lambda x: (x - mean)**2, values)) / n
    return mean, sd, f"{mean:2.5f}±{sd:2.2f}"


def train_model(model, num_epochs, trainloader, criterion, optimizer,
                scheduler, validloader, verbose=False, dump=False):
    """
    Training loop for the torch model
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_losses = []
    valid_losses = []

    avg_train_losses = []
    avg_valid_losses = []

    for _ in tqdm.tqdm(range(num_epochs)):
        epoch_train_loss = []
        epoch_valid_loss = []

        start = time.time()
        # Training
        model.train()
        for graphs, labels in trainloader:

            graphs = graphs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            value = model(graphs)
            loss = criterion(value, labels)
            loss.backward()

            optimizer.step()

            train_losses.append(loss)
            epoch_train_loss.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            for graph, labels in validloader:
                graphs = graph.to(device)
                labels = labels.to(device)
                value = model(graphs)
                loss = criterion(value, labels)
                valid_losses.append(loss)
                epoch_valid_loss.append(loss.item())

        # Scheduler step
        # scheduler.step(valid_loss)
        scheduler.step()

        # Calculate average losses
        avg_train_loss = ploss(epoch_train_loss)[0]
        avg_valid_loss = ploss(epoch_valid_loss)[0]

        avg_train_losses.append(avg_train_loss)
        avg_valid_losses.append(avg_valid_loss)

        if dump:
            with open("./model-outputs/model_results.pkl", "wb") as f:
                print("Saving...")
                pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

        if verbose:
            lr = scheduler.get_last_lr()[0]
            print(
                f"\tT_Loss: {ploss(epoch_train_loss)[-1]} | V_Loss: {ploss(epoch_valid_loss)[-1]}"
                # + f" | T_acc: {avg_train_acc:2.2f} " + "| V_acc: {avg_valid_acc:2.2f}"
                + f" | Time: {time.time()-start:2.2f} | lr: {lr:2.5f}"
            )

    return train_losses, valid_losses, avg_train_losses, avg_valid_losses


class RunResult():

    def __init__(self, batch_size, learning_rate, net_type, n_head, e_size, n_conv):

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.n_type = net_type
        self.n_head = n_head
        self.e_size = e_size
        self.n_conv = n_conv

        self.t_losses = []
        self.v_losses = []
        self.t_epoch_avg_losses = []
        self.v_epoch_avg_losses = []
