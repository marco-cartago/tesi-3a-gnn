import os
import time
import pickle
import tqdm

from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader
import pandas as pd
import h5py
from sklearn.model_selection import KFold

import networks as nets
import data_preprocess_mol as preprocess_mol

IN_EMB_DIM = 19
OUT_EMB_DIM = 1
UPSCALE_DIM = 128

MODELS = {

    # To test a model that just does aggregation
    "JustAggrConvGCNN": (nets.JustAggrConvGCNN,
                         (IN_EMB_DIM, OUT_EMB_DIM, UPSCALE_DIM)),

    # A single convolution with average aggregation
    "OneConvGCNN": (nets.OneConvGCNN,
                    (IN_EMB_DIM, OUT_EMB_DIM, UPSCALE_DIM)),

    # Two convolutions with average aggregation
    "TwoConvGCNN": (nets.TwoConvGCNN,
                    (IN_EMB_DIM, OUT_EMB_DIM, UPSCALE_DIM)),

    # Graph attention network with one head
    "OneLayerOneHeadGAT": (nets.OneLayerOneHeadGAT,
                           (IN_EMB_DIM, OUT_EMB_DIM, UPSCALE_DIM)),

    # Graph attention network with two heads
    "MultiHeadGAT2": (nets.MultiHeadGAT,
                      (IN_EMB_DIM, OUT_EMB_DIM, UPSCALE_DIM, 2)),

    # Graph attention with four heads
    "MultiHeadGAT4": (nets.MultiHeadGAT,
                      (IN_EMB_DIM, OUT_EMB_DIM, UPSCALE_DIM, 4)),
}

MODELS_NCONV_HEADS = {

    # To test a model that just does aggregation
    "JustAggrConvGCNN": (0, 0),

    # A single convolution with average aggregation
    "OneConvGCNN": (1, 0),

    # Two convolutions with average aggregation
    "TwoConvGCNN": (2, 0),

    # Graph attention network with one head
    "OneLayerOneHeadGAT": (1, 1),

    # Graph attention network with two heads
    "MultiHeadGAT2": (2, 2),

    # Graph attention network with four heads
    "MultiHeadGAT4": (2, 4),
}


def ploss(values: list):
    n = len(values)
    mean = sum(values) / n
    sd = sum(map(lambda x: (x - mean)**2, values)) / n
    return mean, sd, f"{mean:2.5f}±{sd:2.2f}"


def kf_train_model(model, fold, num_epochs, trainloader, criterion, optimizer,
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

    for _ in tqdm.tqdm(range(num_epochs), desc=f"Fold {fold:2.0f}"):
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

            train_losses.append(loss.item())
            epoch_train_loss.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            for graph, labels in validloader:
                graphs = graph.to(device)
                labels = labels.to(device)
                value = model(graphs)
                loss = criterion(value, labels)
                valid_losses.append(loss.item())
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

    def __init__(self, fold_num, batch_size, learning_rate, model_name, n_head, e_size, n_conv):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.fold_num = fold_num

        self.model_name = model_name
        self.n_head = n_head
        self.e_size = e_size
        self.n_conv = n_conv

        self.t_losses = []
        self.v_losses = []
        self.t_epoch_avg_losses = []
        self.v_epoch_avg_losses = []


def kfold_validate_models(model_class_name, num_folds=5, restarts=2):

    print(f"Validating {model_class_name}")
    run_result_list = []
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_set)):

        for restart in range(restarts):

            # Split the data based on train/test indices
            train_subset = train_set.iloc[train_idx].reset_index(drop=True)
            valid_subset = train_set.iloc[val_idx].reset_index(drop=True)

            # Load and preprocess the data
            train_subset_dataset = preprocess_mol.H5PyMolDataset(
                h5f, train_subset, "idx", train=True)
            train_dataloader = DataLoader(
                train_subset_dataset, batch_size=BATCH_SIZE,
                shuffle=False, collate_fn=preprocess_mol.graph_collate_fn_h5)

            valid_subset_dataset = preprocess_mol.H5PyMolDataset(
                h5f, valid_subset, "idx", train=True)
            valid_dataloader = DataLoader(
                valid_subset_dataset, batch_size=BATCH_SIZE,
                shuffle=False, collate_fn=preprocess_mol.graph_collate_fn_h5)

            # Initialize model, optimizer, scheduler
            model_class, args = MODELS[model_class_name]
            model_istance = model_class(*args)

            # Define criterion, optimizer and scheduler
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(
                model_istance.parameters(),
                lr=START_LEARNING_RATE
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=STEP_SIZE,
                gamma=0.80,
            )

            # Train the model
            train_losses, valid_losses, avg_train_losses, avg_valid_losses = \
                kf_train_model(model_istance,
                               fold,
                               num_epochs=NUM_EPOCHS,
                               trainloader=train_dataloader,
                               validloader=valid_dataloader,
                               criterion=criterion,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               verbose=True,
                               dump=True)

            # Store and process the results
            n_conv, n_heads = MODELS_NCONV_HEADS[model_class_name]

            run_result = RunResult(fold_num=fold,
                                   batch_size=BATCH_SIZE,
                                   learning_rate=START_LEARNING_RATE,
                                   model_name=model_class_name,
                                   n_head=n_conv,
                                   e_size=UPSCALE_DIM,
                                   n_conv=n_heads)

            run_result.t_losses = train_losses
            run_result.v_losses = valid_losses
            run_result.t_epoch_avg_losses = avg_train_losses
            run_result.v_epoch_avg_losses = avg_valid_losses
            run_result_list.append(run_result)

    return run_result_list


load_dotenv()

# Set arguments
mol_data_path = "./mol-data/mol_sample_metadata.csv"
metadata = pd.read_csv(mol_data_path)

# Path to the h5py file containing all the preprocessed embeddings, labels
# and adjacency matrix
treated_molecules_path = "./mol_dataset.h5"
if treated_molecules_path is None:
    raise ValueError("Invalid .h5 data path.")

h5f = h5py.File(treated_molecules_path, "r")

train_set: pd.DataFrame = metadata[metadata["set"] == "train"]

NUM_EPOCHS = 100
BATCH_SIZE = 32
START_LEARNING_RATE = 0.0001
STEP_SIZE = 8


def main():

    print(MODELS.items())
    obj = []

    for cl_name in MODELS.keys():
        obj.append(kfold_validate_models(cl_name))

        with open("./model-outputs/out_valid.pkl", "wb") as f:
            pickle.dump(obj, f)


if __name__ == "__main__":
    main()
