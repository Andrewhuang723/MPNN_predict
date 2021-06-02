import torch
import dgl
from dgl import DGLGraph
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDConfig
import os

device = torch.device("cuda")

gdb_dir = './data/GDB9Quantum.csv'
save_file = './data/graphs_explicit_H.bin'
smiles = pd.read_csv(gdb_dir, index_col=None)['SMILES']

ELEM_LIST = ['H', 'C', 'N', 'O', 'F']

def get_mol(smiles, explicit_H=False):
    if explicit_H is True:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    else:
        mol = Chem.MolFromSmiles(smiles)

    Chem.Kekulize(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    return mol

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

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
            + onek_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, "None"])
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


def mol2graph(smiles, explicit_H=False, block=False):
    mol = get_mol(smiles, explicit_H)
    n_atoms = mol.GetNumAtoms()
    g = DGLGraph()
    node_feats = []

    if block is True:
        MolBlock = Chem.MolToMolBlock(mol)
        mol_2 = Chem.MolFromMolBlock(MolBlock)
        position = mol_2.GetConformers()[0].GetPositions()

    else:
        position = mol.GetConformers()[0].GetPositions()

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
    g.edata['h'] = torch.Tensor([a.tolist() for a in edge_feats]).squeeze(0)
    return g

# Converting all SMILES into graphs
if __name__ == '__main__':
    if not os.path.exists(save_file):
        G = []
        for i in range(len(smiles)):
            try:
                graph = mol2graph(smiles[i], explicit_H=True)
                G.append(graph)
            except:
                print(i)
                graph = mol2graph(smiles[i], block=True)
                G.append(graph)
                pass
        dgl.data.save_graphs(save_file, G)
        print('\n----------------------------')
        print('graphs save done')
