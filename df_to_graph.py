"""
Form a graph data structure (Pytorch geometric) according to the input data frame.
Rule: Each atom represents a node. If the distance between two atoms are less than or 
equal to 4.5 Angstrom (may become a tunable hyper-parameter in the future), then an 
undirected edge is formed between these two atoms.

Input:
atoms: dataframe containing the 3-d coordinates of atoms

Output:
A Pytorch-gemometric graph data with following contents:
    - node_attr (Pytorch Tensor): Node feature matrix with shape [num_nodes, num_node_features]. e.g.,
      x = torch.tensor([[-1], [0], [1]], dtype=torch.float
    - edge_index (Pytorch LongTensor): Graph connectivity in COO format with shape [2, num_edges*2]. e.g.,
      edge_index = torch.tensor([[0, 1, 1, 2],
                                 [1, 0, 2, 1]], dtype=torch.long)

Forming the final output graph:
    data = Data(x=x, edge_index=edge_index)
"""

import numpy as np 
from biopandas.mol2 import PandasMol2
import pandas as pd

if __name__ == "__main__":
    pd.options.display.max_rows = 999
    hydrophobicity = {'ALA':1.8,'ARG':-4.5,'ASN':-3.5,'ASP':-3.5,
                      'CYS':2.5,'GLN':-3.5,'GLU':-3.5,'GLY':-0.4,
                      'HIS':-3.2,'ILE':4.5,'LEU':3.8,'LYS':-3.9,
                      'MET':1.9,'PHE':2.8,'PRO':-1.6,'SER':-0.8,
                      'THR':-0.7,'TRP':-0.9,'TYR':-1.3,'VAL':4.2}
    
    binding_probability = {'ALA':0.701,'ARG':0.916,'ASN':0.811,'ASP':1.015,
                           'CYS':1.650,'GLN':0.669,'GLU':0.956,'GLY':0.788,
                           'HIS':2.286,'ILE':1.006,'LEU':1.045,'LYS':0.468,
                           'MET':1.894,'PHE':1.952,'PRO':0.212,'SER':0.883,
                           'THR':0.730,'TRP':3.084,'TYR':1.672,'VAL':0.884}

    mol_path = './data/1agrB01-1.mol2'
    atoms = PandasMol2().read_mol2(mol_path)
    atoms = atoms.df[['atom_id','subst_name', 'atom_type', 'atom_name', 'x', 'y', 'z', 'charge']]
    atoms['residue'] = atoms['subst_name'].apply(lambda x: x[0:3])
    atoms['hydrophobicity'] = atoms['residue'].apply(lambda x: hydrophobicity[x])
    atoms['binding_probability'] = atoms['residue'].apply(lambda x: binding_probability[x])
    atoms = atoms[['atom_type', 'residue', 'x', 'y', 'z', 'charge', 'hydrophobicity', 'binding_probability']]
    print(atoms)

    # calculate the pair-wise distances and forming edges