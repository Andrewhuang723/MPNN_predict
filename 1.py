import torch
import dgl
import PIL.Image as image
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from sklearn.model_selection import train_test_split
from dgllife.utils.eval import Meter

device = torch.device("cuda")

data = pd.read_csv("./data/GDB9Quantum.csv")
data1 = data.drop(["Index", "SMILES", "SMILES from B3LYP relaxation"], axis=1)
properties = data1.values

print(properties.shape)
mean = np.mean(properties, axis=0)
mean = mean[np.newaxis,:]
std = np.std(properties, axis=0)
std = std[np.newaxis,:]
standardize_values = (properties - mean)/std
standardize_values = torch.tensor(standardize_values)

a = []
for j in range(standardize_values.shape[1]):
    for i in range(standardize_values.shape[0]):
        if torch.abs(standardize_values[i, j]) > 3:
            a.append(i)

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

a = unique(a)

data = pd.read_csv("./data/compounds.csv", index_col=None, header=None).values.reshape(-1)

actual_y = []
other = []
actual_X = []
for i in range(len(properties)):
    if i not in a:
        actual_y.append(properties[i])
        actual_X.append(data[i])
actual_y = np.array(actual_y)
other = np.array(other)
print(actual_y.shape)

mu = np.mean(actual_y, axis=0)
mu = mu[np.newaxis,:]
sd = np.std(actual_y, axis=0)
sd = sd[np.newaxis,:]
np.savez("stat.npz", mean=mu, std=sd)
actual_y = (actual_y - mu) / sd
actual_y = torch.tensor(actual_y)

actual_X = np.array(actual_X)
X_train, X_test = train_test_split(actual_X, test_size=0.2, random_state=0)
y_train, y_test = train_test_split(actual_y, test_size=0.2, random_state=0)
torch.save(y_train, "./data/train_regression.pkl")
torch.save(y_test, "./data/test_regression.pkl")
print("X_train: ", X_train.shape, "\ny_train: ", y_train.shape)

ELEM_LIST = ['H', 'C', 'N', 'O', 'F']

def get_mol(smiles, explicit_H=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    return mol
print(X_train[103])
# mol_0 = Chem.MolFromSmiles(X_train[103])
# img = Chem.Draw.MolToImage(mol_0)
# print(image._show(img))

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

import os
from rdkit import RDConfig

def donor_acceptor(mol):
    fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    mol_feat = fdef.GetFeaturesForMol(mol)
    idx = [atom_feat.GetAtomIds()[0] for atom_feat in mol_feat]
    mol_idx = [atom.GetIdx() for atom in mol.GetAtoms()]
    k = list(range(mol.GetNumAtoms()))
    a = [i for i in mol_idx if i not in idx]
    for i, atom_feat in zip(idx, mol_feat):
        k[i] = atom_feat.GetFamily()
    for i in a:
        k[i] = "None"
    return k


def atom_features(atom, position, i, mol):
    x, y, z = position[i]
    return (torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
            + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
            + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])
            + onek_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2])
            + onek_encoding_unk(donor_acceptor(mol)[i], ["Donor", "Acceptor", "None"])
            + [atom.GetIsAromatic()]
            + [atom.GetAtomicNum()]
            + [atom.GetTotalNumHs()]))
            #+ [x, y, z]))

def getbondlength(atom_i, atom_j, position):
    a = np.sqrt(np.sum(np.square(position[atom_i] - position[atom_j])))
    bin = [np.clip(a, 0, 1.0), np.clip(a, 1.0, 1.2), np.clip(a, 1.2, 1.2501), np.clip(a, 1.2501, 1.3001), np.clip(a, 1.3001, 1.3501),
           np.clip(a, 1.3501, 1.4001), np.clip(a, 1.4001, 1.4501), np.clip(a, 1.4501, 1.5001), np.clip(a, 1.5001, 1.5501), np.clip(a, 1.5501, 1.60), np.clip(a, 1.60, 2),
           np.clip(a, 2, 2.5), np.clip(a, 2.5, 3), np.clip(a, 3, 3.5), np.clip(a, 3.5, np.Inf)]
    idx = [a == i for i in bin]
    return idx

def getbondtype(bond):
    return onek_encoding_unk(bond.GetBondType(),
            [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])


from dgllife.utils import CanonicalBondFeaturizer

def mol2graph(smiles, explicit_H=False):
    mol = get_mol(smiles, explicit_H)
    n_atoms = mol.GetNumAtoms()
    g = DGLGraph()
    node_feats = []

    MolBlock = Chem.MolToMolBlock(mol)
    if explicit_H is True:
        mol_2 = Chem.AddHs(Chem.MolFromMolBlock(MolBlock))
    else:
        mol_2 = Chem.MolFromMolBlock(MolBlock)
    position = mol_2.GetConformers()[0].GetPositions()

    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        node_feats.append(atom_features(atom, position, i, mol))

    g.add_nodes(n_atoms)

    bond_src = []
    bond_dst = []
    edge_feats = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bondlength = torch.Tensor([getbondlength(u, v, position), getbondlength(u, v, position)])
        bondtype = torch.Tensor([getbondtype(bond), getbondtype(bond)])
        b_f = torch.cat([bondlength, bondtype], dim=-1)
        edge_feats.extend(b_f)
        bond_src.extend([u, v])
        bond_dst.extend([v, u])
    g.add_edges(bond_src, bond_dst)

    g.ndata['h'] = torch.Tensor([a.tolist() for a in node_feats])
    g.edata["h"] = torch.Tensor([a.tolist() for a in edge_feats]).squeeze(0)
    return g

SAMPLE_MOL = mol2graph(X_train[103], explicit_H=False)

print(SAMPLE_MOL, SAMPLE_MOL.ndata["h"].shape, SAMPLE_MOL.edata["h"].shape)

import networkx as nx
import matplotlib.pyplot as plt

nx.draw(SAMPLE_MOL.to_networkx(), with_labels=True)
plt.show()


# train_graphs = []
# for i in range(len(X_train)):
#     try:
#         graph = mol2graph(X_train[i], explicit_H=True)
#         train_graphs.append(graph)
#     except:
#         pass
# dgl.data.save_graphs("./data/train_graphs_explicit_H.bin", train_graphs)
# print("train_graphs save done")

test_graphs = []
print(len(X_test))
for i in range(len(X_test)):
    try:
        graph = mol2graph(X_test[i], explicit_H=True)
        test_graphs.append(graph)
    except:
        pass
dgl.data.save_graphs("./data/test_graphs_explicit_H.bin", test_graphs)
print("test_graphs save done")