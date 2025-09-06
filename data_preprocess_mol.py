"""
In this module are contained all the functions and classes responsible for 
creating the h5df file containing the dataset.

The structure is
/(train or test)/(adj or emb or lab)/molecule_id
 - (train or test) the distintion of the data set we are currently in
 - (adj or emb or lab) each of these sections saves in the corresponding a dataset named molecule_id that saves 
  - the adjacency matrix
  - the node embeddings
  - the molecule label
"""

import numpy as np
import mne
import networkx as nx
import pandas as pd
import pickle
import h5py
import time
import os

from torch.utils.data import Dataset
from scipy import signal, stats
from layers import *


class H5PyMolDataset(Dataset):

    def __init__(self, h5_database, moldata: pd.DataFrame, idx_col: str, train: bool = True):

        self.h5_database = h5_database
        self.moldata: pd.DataFrame = moldata
        self.idx: str = idx_col
        self.set = "train" if train else "test"

    def __len__(self):
        return (self.moldata.shape)[0]

    def load_data_from_h5py(self, idx: int):
        """Load data from the HDF5 file."""
        dataset_name = f"molecule_{idx}"
        base = self.h5_database[self.set]
        adj = base["adjs"][dataset_name][:]
        emb = base["embs"][dataset_name][:]
        lab = base["labs"][dataset_name][()]
        res = ((adj.astype(np.float32),
                emb.astype(np.float32)),
               lab.astype(np.float32))
        return res

    def __getitem__(self, index: int):
        local_idx = self.moldata[self.idx][index]
        res = self.load_data_from_h5py(local_idx)
        return res


def to_torch(data: tuple, etype="bands") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepares `data` for `combine_simple_graphs` casting to a `torch.Tensor`
    """

    numpy_adj, numpy_emb = data
    emb = torch.from_numpy(numpy_emb)  # inter jon (da parte di Simone)
    adj = torch.from_numpy(numpy_adj)
    return (adj, emb)


def combine_simple_graphs_h5(graphs_adjs: list[torch.Tensor], embeddings: list) -> EnrichedGraph:
    """
    Given a list of graph adjacency matrices returns a `EnrichedGraph`. It 
    creates a graph that has each graph `G_i` as a indipendent unconnected 
    component. 
    """

    # Get number of vertices for each graph
    v_num = [int(g.size(0)) for g in graphs_adjs]
    total_vertices = sum(v_num)
    # Initialize correspondence array
    correspondence = torch.zeros(total_vertices, dtype=torch.int32)
    # Fill correspondence array
    pos = 0
    for i, n in enumerate(v_num):
        for k in range(pos, pos + n):
            correspondence[k] = i
        pos += n

    adjacency_matrix = torch.block_diag(*graphs_adjs)
    # Combine embeddings
    combined_embeddings = torch.vstack(embeddings)
    res = EnrichedGraph(
        adj_mat=adjacency_matrix, emb_mat=combined_embeddings, corrisp=correspondence
    )
    return res


def gen_labels_h5(labels) -> torch.Tensor:
    """
    Starting from the individual single values return the label vector of
    the specified size.
    """

    return torch.stack(
        list(
            map(
                lambda x: torch.from_numpy(np.array(x, dtype=np.float32)),
                labels)
        )
    ).reshape(len(labels), 1)  # * 313.754 # To convert in kcal/mol.


def graph_collate_fn_h5(batch):

    # Unpacking
    raw_graphs, labels = zip(*batch)

    # Generate
    pairs = [(lambda x: to_torch(x))(s)
             for s in raw_graphs]  # Convert to torch format
    adjs = [(lambda x: x[0])(p) for p in pairs]              # Obtain adjs
    embeddings = [(lambda x: x[1])(p) for p in pairs]        # Obtain labels
    # Generate label vector
    tensor_labels = gen_labels_h5(labels)

    # Package
    enriched_graph_batch = combine_simple_graphs_h5(adjs, embeddings)

    # Send :)
    return enriched_graph_batch, tensor_labels
