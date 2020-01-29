import argparse
import numpy as np 
import pandas as pd 
import os
from biopandas.mol2 import PandasMol2
#from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Dataset


"""
Questions before starting: 
 1. How to represent graph data structure? NetworkX or Pytorch geometric data?
    - use NextworkX first then convert to Pytorch geometric data format.
 2. Pandas or BioPandas?
    - BioPandas
 3. Batch?
 4. Normalize features?
"""


def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-op',
                        default='control_vs_heme',
                        required=False,
                        choices = ['control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'],
                        help="'control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'")
    parser.add_argument('-root_dir',
                        default='../MolNet-data/',
                        required=False,
                        help='directory to load data for 5-fold cross-validation.')   
    parser.add_argument('-result_file_suffix',
                        default='default_run',
                        required=False,
                        help='suffix to result file')                        
    parser.add_argument('-batch_size',
                        type=int,
                        default=32,
                        required=False,
                        help='the batch size, normally 2^n.')
    parser.add_argument('-num_control',
                        default=74784,
                        required=False,
                        help='number of control data points, used to calculate the positive weight for the loss function')
    parser.add_argument('-num_heme',
                        default=22944,
                        required=False,
                        help='number of heme data points, used to calculate the positive weight for the loss function.')
    parser.add_argument('-num_nucleotide',
                        default=59664,
                        required=False,
                        help='number of num_nucleotide data points, used to calculate the positive weight for the loss function.')
    return parser.parse_args()


class MolDatasetCV(Dataset):
    """
    Dataset for MolNet, can be used to load multiple folds for training or single fold for validation and testing
    """
    def __init__(self, op, root_dir, folds, transform=None):
        """
        Args:
            op: operation mode, heme_vs_nucleotide, control_vs_heme or control_vs_nucleotide. 
            root_dir: folder containing all mol files.
            folds: a list containing folds to generate training data, for example [1,2,3,4], [5].
            transform: transform to be applied to graphs.
        """
        self.op = op
        self.root_dir = root_dir
        self.folds = folds
        self.transform = transform
        self.hydrophobicity = {'ALA':1.8,'ARG':-4.5,'ASN':-3.5,'ASP':-3.5,
                               'CYS':2.5,'GLN':-3.5,'GLU':-3.5,'GLY':-0.4,
                               'HIS':-3.2,'ILE':4.5,'LEU':3.8,'LYS':-3.9,
                               'MET':1.9,'PHE':2.8,'PRO':-1.6,'SER':-0.8,
                               'THR':-0.7,'TRP':-0.9,'TYR':-1.3,'VAL':4.2}
        self.binding_probability = {'ALA':0.701,'ARG':0.916,'ASN':0.811,'ASP':1.015,
                                    'CYS':1.650,'GLN':0.669,'GLU':0.956,'GLY':0.788,
                                    'HIS':2.286,'ILE':1.006,'LEU':1.045,'LYS':0.468,
                                    'MET':1.894,'PHE':1.952,'PRO':0.212,'SER':0.883,
                                    'THR':0.730,'TRP':3.084,'TYR':1.672,'VAL':0.884}
        print('--------------------------------------------------------')
        if len(folds) == 1:
            print('generating validation dataset...')
        elif len(folds) == 4:
            print('generating training dataset...')
        print('folds: ', self.folds)
        
        self.pocket_classes = self.op.split('_vs_') # two classes of binding sites
        assert(len(self.pocket_classes)==2)
        self.class_name_to_int = {}
        for i in range(len(self.pocket_classes)):
            self.class_name_to_int[self.pocket_classes[i]] = i
        print('class name to integer map: ', self.class_name_to_int)

        self.folder_dirs = [] # directory of folders containing training images                                 
        self.folder_classes = [] # list of classes of each folder, integer        
        for pocket_class in self.pocket_classes:
            for fold in self.folds:
                self.folder_classes.append(self.class_name_to_int[pocket_class])
                self.folder_dirs.append(self.root_dir + pocket_class + '/fold_' + str(fold))
        print('getting training data from following folders: ', self.folder_dirs)
        print('folder classes: ', self.folder_classes)

        self.list_of_files_list = [] # directory of files for all folds, [[files for fold 1], [files for fold 2], ...]
        for folder_dir in self.folder_dirs:
            self.list_of_files_list.append([os.path.join(folder_dir, name) for name in os.listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, name))])
        #print(self.list_of_files_list)

        self.folder_dirs_lengths = [] # lengths of the folders
        for files_list in self.list_of_files_list:
            self.folder_dirs_lengths.append(len(files_list))
        print('lengths of the folders: ', self.folder_dirs_lengths)

    def __len__(self):
        return sum(self.folder_dirs_lengths)

    def __getitem__(self, idx):
        """
        Built-in function to retrieve item(s) in the dataset.
        Args:
            idx: index of the image
        """
        # get dataframe directory
        folder_idx, sub_idx = self.__locate_file(idx)
        mol_dir = self.list_of_files_list[folder_idx][sub_idx]
        
        print('mol file to read: ', mol_dir)
        # read dataframe
        mol_df = self.__read_mol(mol_dir)

        # get label 
        label = self.__get_class_int(folder_idx)

        # apply transform to PIL image if applicable
        #if self.transform:
        #    image = self.transform(image)

        return mol_df, label

    def __locate_file(self, idx):
        """
        Function to locate the directory of file
        Args:
            idx: an integer which is the index of the file.
        """
        low = 0
        up = 0
        for i in range(len(self.folder_dirs_lengths)):
            up += self.folder_dirs_lengths[i]
            if idx >= low and idx <up:
                sub_idx = idx - low
                #print('folder:', i)
                #print('sub_idx:', sub_idx)
                #print('low:', low)
                #print('up', up)
                return i, sub_idx
            low = up  

    def __get_class_int(self, folder_idx):
        """
        Function to get the label of an image as integer
        """
        return self.folder_classes[folder_idx]


    def __read_mol(self, mol_path):
        """
        Read the mol2 file as a dataframe.
        """
        atoms = PandasMol2().read_mol2(mol_path)
        atoms = atoms.df[['atom_id','subst_name', 'atom_type', 'atom_name', 'x', 'y', 'z', 'charge']]
        atoms['residue'] = atoms['subst_name'].apply(lambda x: x[0:3])
        atoms['hydrophobicity'] = atoms['residue'].apply(lambda x: self.hydrophobicity[x])
        atoms['binding_probability'] = atoms['residue'].apply(lambda x: self.binding_probability[x])
        atoms = atoms[['atom_type', 'residue', 'x', 'y', 'z', 'charge', 'hydrophobicity', 'binding_probability']]
        return atoms

    def __form_graph(self, atoms):
        """
        Form a graph data structure (Pytorch geometric) according to the input data frame.
        Rule: Each atom represents a node. If the distance between two atoms are less than or 
        equal to 4.5 Angstrom (may become a tunable hyper-parameter in the future), then an 
        undirected edge is formed between these two atoms. 

        Input:
        atoms: dataframe containing the 3-d coordinates of atoms.

        Output:
        A Pytorch-gemometric graph data with following contents:
            - node_attr (Pytorch Tensor): Node feature matrix with shape [num_nodes, num_node_features].
            - edge_index (Pytorch LongTensor): Graph connectivity in COO format with shape [2, num_edges].
        """


if __name__ == "__main__":
    pd.options.display.max_rows = 999
    args = get_args()
    op = args.op    
    root_dir = args.root_dir
    result_file_suffix = args.result_file_suffix
    batch_size = args.batch_size
    print('data directory:', root_dir)
    print('batch size: ', batch_size)
    num_control = args.num_control
    num_heme = args.num_heme
    num_nucleotide = args.num_nucleotide

    training_folds = [1,2,3,4]
    val_fold = 5
    #training_set = BionoiDatasetCV(op=op, root_dir=root_dir, folds=training_folds)
    val_set = MolDatasetCV(op=op, root_dir=root_dir, folds=[val_fold])
    print(val_set[0])
    