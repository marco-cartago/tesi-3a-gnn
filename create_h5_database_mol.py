import mne
import pandas as pd
import os
import tqdm
import h5py
import numpy as np

import networkx as nx
from rdkit import Chem

from dotenv import load_dotenv

ELEMENT_SYMBOLS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
    '*'  # The element of surprise
]


def graph_from_smiles(smiles: str, global_node: bool = True):
    """Convert SMILES string to RDKit molecule object and NetworkX graph"""

    # Parse SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Create NetworkX graph
    G = nx.Graph()

    # Add atoms as nodes
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   hybridization=atom.GetHybridization(),
                   aromatic=atom.GetIsAromatic()
                   )

    # Add bonds as edges
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType(),
                   aromatic=bond.GetIsAromatic()
                   )

    # Add central node
    if global_node:
        existing_nodes = list(G.nodes())
        curr_node = len(existing_nodes)
        G.add_node(curr_node,
                   symbol='*',
                   atomic_num=0,
                   formal_charge=0,
                   hybridization=False,
                   aromatic=False
                   )
        G.add_edges_from([(curr_node, node) for node in existing_nodes])

    # Add self loops
    G.add_edges_from([(node, node) for node in G.nodes()])
    adj_mat = nx.to_numpy_array(G)

    return G, adj_mat


def generate_graph_embeddings(g: nx.Graph) -> np.ndarray:
    node_num = g.number_of_nodes()
    o_features = ['atomic_num', 'formal_charge', 'aromatic']
    nf = len(o_features)
    hybridization_types = ['sp', 'sp2', 'sp3', 'sp2d',
                           'sp3d', 'sp3d2', 'sp3d3', 'sp3d4', 'sp3d5']
    h_size = len(hybridization_types)
    CHNOPS = ['C', 'H', 'N', 'O', 'P', 'S']
    chnops_size = len(CHNOPS)

    total_features = 1 + chnops_size + nf + h_size
    emb = np.empty((total_features, node_num))

    for i, key in enumerate(g.nodes.keys()):
        ddict = g.nodes[key]

        if 'symbol' in ddict.keys():
            emb[0, i] = float(ELEMENT_SYMBOLS.index(
                g.nodes[key]['symbol']) + 1)
        else:
            emb[0, i] = 0

        # One-hot encoding per CHNOPS
        symbol = ddict.get('symbol', '')
        for j, element in enumerate(CHNOPS):
            if symbol == element:
                emb[1 + j, i] = 1.0
            else:
                emb[1 + j, i] = 0.0

        # Other features
        for j in range(nf):
            feature = o_features[j]
            feature_idx = 1 + chnops_size + j
            if feature in ddict.keys():
                emb[feature_idx, i] = float(ddict[feature])
            else:
                emb[feature_idx, i] = 0

        # One-hot encoding for hybridization type
        hybridization = ddict.get('hybridization', '')
        for j, h_type in enumerate(hybridization_types):
            if hybridization == h_type:
                emb[1 + chnops_size + nf + j, i] = 1.0
            else:
                emb[1 + chnops_size + nf + j, i] = 0.0

    return emb.T


def label_tensor(l, round_digits: int = 8) -> np.ndarray:
    n = len(l)
    out_dim = 1
    outp = np.zeros((n, out_dim))
    for i in range(n):
        outp[i, 0] = float(l[i])

    return outp


def process_molecule(output_h5py_file, metadata, molecule: str,
                     energy: float, group_name: str, idx: int, verbose=False):

    # Retrive moelcule and connection structure
    mol_g, adj = graph_from_smiles(smiles=molecule, global_node=True)
    embs = generate_graph_embeddings(mol_g)

    # Create group for the file (if not present)
    if group_name not in output_h5py_file:
        output_h5py_file.create_group(group_name)

    dataset_at_group = output_h5py_file[group_name]

    # Not found keys problems
    if 'adjs' not in dataset_at_group:
        dataset_at_group.create_group('adjs')

    if 'embs' not in dataset_at_group:
        dataset_at_group.create_group('embs')

    if 'labs' not in dataset_at_group:
        dataset_at_group.create_group('labs')

    # Save data array
    dataset_name = f"molecule_{idx}"
    dataset_at_group["adjs"].create_dataset(dataset_name, data=adj)
    dataset_at_group["embs"].create_dataset(dataset_name, data=embs)
    dataset_at_group["labs"].create_dataset(
        dataset_name, data=np.array(energy))

    metadata.append([idx, molecule, group_name])


def initialize_h5py_file(output_h5py_file):
    """Initialize the HDF5 file with top-level groups."""
    with h5py.File(output_h5py_file, "w") as h5f:
        if "train" not in h5f:
            h5f.create_group("train")
        if "test" not in h5f:
            h5f.create_group("test")

        for g in ["train", "test"]:
            h5f[g].create_group("adjs")
            h5f[g].create_group("embs")
            h5f[g].create_group("labs")


def main():

    # Loading .env
    load_dotenv()

    mol_path = "./mol-data/MyRoboBohr2.csv"
    mol_info = pd.read_csv(mol_path)
    print("Dataset: \n", mol_info)

    metadata = []

    # Path to the .h5 file to create
    db_path = "./mol_dataset.h5"

    # train/validation/test split
    mol_info = mol_info.sample(
        frac=1, random_state=0, replace=False).reset_index(drop=True)

    len_mol_info = len(mol_info)
    n_train_samples = int(0.8 * len_mol_info)
    train_or_test = list(
        map(
            lambda x: "train" if x < n_train_samples else "test",
            [i for i in range(len_mol_info)]
        )
    )

    # Initalize and open the file
    initialize_h5py_file(db_path)
    output_h5py_file = h5py.File(db_path, "w")

    idx = mol_info["id"]
    smiles = mol_info["Canonical SMILES"]
    eat = mol_info["Eat"]

    # Populate the h5py file
    for i in range(len(smiles)):
        print(smiles[i], eat[i], train_or_test[i], i)
        process_molecule(output_h5py_file, metadata,
                         smiles[i], eat[i], train_or_test[i], idx[i])

    # Save metadata
    metadata = pd.DataFrame(metadata, columns=["idx", "smiles", "set"])
    metadata.to_csv("./mol_sample_metadata.csv", index_label=False)

    # Close the file
    output_h5py_file.close()


if __name__ == "__main__":
    main()
