import argparse
import numpy as np 
import pandas as pd 
import os
from biopandas.mol2 import PandasMol2
#from torch.utils.data import Dataset, DataLoader
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data import DataLoader
from scipy.spatial import distance
import statistics


"""
Questions before starting: 
 1. How to represent graph data structure? NetworkX or Pytorch geometric data?
    - Don't need to manipulate the graph, so the dataset should directly output Pytorch geometric data.
 2. Pandas or BioPandas?
    - BioPandas
 3. Batch?
    - To implement batch, just use DataLoader from Pytorch geometric to wrap the datasets object. However, whether to 
      use mini-batch data is not determined yet.
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
                        default='../MolNet-data/pockets/',
                        required=False,
                        help='directory to load mol2 data for 5-fold cross-validation.')

    parser.add_argument('-pop_dir',
                        default='../MolNet-data/pops/',
                        required=False,
                        help='directory to load sasa data.')

    parser.add_argument('-profile_dir',
                        default='../MolNet-data/profiles/',
                        required=False,
                        help='directory to load sasa data.')                                               
    
    parser.add_argument('-result_file_suffix',
                        default='default_run',
                        required=False,
                        help='suffix to result file')                        
    
    parser.add_argument('-batch_size',
                        type=int,
                        default=1,
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
    def __init__(self, op, root_dir, pop_dir, profile_dir, folds, threshold, features_to_use, transform=None):
        """
        Args:
            op: operation mode, heme_vs_nucleotide, control_vs_heme or control_vs_nucleotide. 
            root_dir: folder containing all mol files.
            folds: a list containing folds to generate training data, for example [1,2,3,4], [5].
            threshold: thresh hold of the distance between atoms to form an edge.
            features_to_use: list of features to use for deep learning. Should be subset of ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sasa', 'sequence_entropy']
            transform: transform to be applied to graphs.
        """
        self.op = op
        self.root_dir = root_dir
        self.pop_dir = pop_dir
        self.profile_dir = profile_dir
        self.folds = folds
        self.threshold = threshold
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
        total_features = ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sasa', 'sequence_entropy']
        assert(set(features_to_use).issubset(set(total_features))) # features to use should be subset of ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sasa', 'sequence_entropy']
        self.features_to_use = features_to_use
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

        self.folder_dirs = [] # directories of folders containing training data                                 
        self.pop_sub_dirs = [] # directories of folders containing pop data
        self.profile_sub_dirs = [] # directories of folders containing profile data
        self.folder_classes = [] # list of classes of each folder, integer    
        for pocket_class in self.pocket_classes:
            for fold in self.folds:
                self.folder_classes.append(self.class_name_to_int[pocket_class])
                self.folder_dirs.append(self.root_dir + pocket_class + '/fold_' + str(fold))
                self.pop_sub_dirs.append(self.pop_dir + pocket_class + '/fold_' + str(fold))
                self.profile_sub_dirs.append(self.profile_dir + pocket_class + '/fold_' + str(fold))
        print('getting data from following folders: ', self.folder_dirs)
        print('getting pops from following folders: ', self.pop_sub_dirs)
        print('getting profiles from following folders: ', self.profile_sub_dirs)
        print('folder classes: ', self.folder_classes)

        self.list_of_files_list = [] # directory of files for all folds, [[files for fold 1], [files for fold 2], ...]
        for folder_dir in self.folder_dirs:
            self.list_of_files_list.append([os.path.join(folder_dir, name) for name in os.listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, name))])
        #print(self.list_of_files_list)

        self.list_of_pop_files_list = [] # directory of pop files for all folds, [[files for fold 1], [files for fold 2], ...]
        for file_list in self.list_of_files_list:
            pop_files = [(lambda x: self.pop_dir + x.split('pockets/')[-1][:-9] + '.out')(name) for name in file_list]
            self.list_of_pop_files_list.append(pop_files)
        #print(self.list_of_pop_files_list)

        self.list_of_profile_files_list = [] # directory of profile files for all folds, [[files for fold 1], [files for fold 2], ...]
        for file_list in self.list_of_files_list:
            profile_files = [(lambda x: self.profile_dir + x.split('pockets/')[-1][:-9] + '.profile')(name) for name in file_list]
            self.list_of_profile_files_list.append(profile_files)
        #print(self.list_of_profile_files_list)

        self.folder_dirs_lengths = [] # lengths of the folders
        for files_list in self.list_of_files_list:
            self.folder_dirs_lengths.append(len(files_list))
        print('lengths of the folders: ', self.folder_dirs_lengths)
        print('--------------------------------------------------------')

    def __len__(self):
        return sum(self.folder_dirs_lengths)

    def __getitem__(self, idx):
        """
        Built-in function to retrieve item(s) in the dataset.
        Args:
            idx: index of the data
        """
        folder_idx, sub_idx = self.__locate_file(idx) # get dataframe directory
        mol_dir = self.list_of_files_list[folder_idx][sub_idx] # get dataframe directory
        pop_dir = self.list_of_pop_files_list[folder_idx][sub_idx]
        profile_dir = self.list_of_profile_files_list[folder_idx][sub_idx]
        label = self.__get_class_int(folder_idx) # get label 
        #print('pocket dir:', mol_dir)
        #print('pop dir:', pop_dir)
        #print('profile dir', profile_dir)
        graph_data = self.__read_mol(mol_dir, pop_dir, profile_dir, label) # read dataframe as pytorch-geometric graph data

        ''' apply transform to PIL data if applicable '''
        #if self.transform:
        #    mol = self.transform(mol)
        return graph_data

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
                return i, sub_idx
            low = up  

    def __get_class_int(self, folder_idx):
        """
        Function to get the label of an image as integer
        """
        return self.folder_classes[folder_idx]

    def __read_mol(self, mol_path, pop_path, profile_path, label):
        """
        Read the mol2 file as a dataframe.
        """
        atoms = PandasMol2().read_mol2(mol_path)
        atoms = atoms.df[['atom_id','subst_name', 'atom_type', 'atom_name', 'x', 'y', 'z', 'charge']]
        atoms['residue'] = atoms['subst_name'].apply(lambda x: x[0:3])
        atoms['hydrophobicity'] = atoms['residue'].apply(lambda x: self.hydrophobicity[x])
        atoms['binding_probability'] = atoms['residue'].apply(lambda x: self.binding_probability[x])
        center_distances = self.__compute_dist_to_center(atoms[['x','y','z']].to_numpy())
        atoms['distance_to_center'] = center_distances
        siteresidue_list = atoms['subst_name'].tolist()
        qsasa_data = self.__extract_sasa_data(siteresidue_list, pop_path)
        atoms['sasa'] = qsasa_data
        seq_entropy_data = self.__extract_seq_entropy_data(siteresidue_list, profile_path) # sequence entropy data with subst_name as keys
        atoms['sequence_entropy'] = atoms['subst_name'].apply(lambda x: seq_entropy_data[x])
        atoms_graph = self.__form_graph(atoms, self.threshold, label)
        return atoms_graph

    def __form_graph(self, atoms, threshold, label):
        """
        Form a graph data structure (Pytorch geometric) according to the input data frame.
        Rule: Each atom represents a node. If the distance between two atoms are less than or 
        equal to 4.5 Angstrom (may become a tunable hyper-parameter in the future), then an 
        undirected edge is formed between these two atoms. 

        Input:
        atoms: dataframe containing the 3-d coordinates of atoms.
        threshold: distance threshold to form the edge (chemical bond).
        label: class of the data point.

        Output:
        A Pytorch-gemometric graph data with following contents:
            - node_attr (Pytorch Tensor): Node feature matrix with shape [num_nodes, num_node_features]. e.g.,
              x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

            - edge_index (Pytorch LongTensor): Graph connectivity in COO format with shape [2, num_edges*2]. e.g.,
              edge_index = torch.tensor([[0, 1, 1, 2],
                                         [1, 0, 2, 1]], dtype=torch.long)
        
        Forming the final output graph:
            data = Data(x=x, edge_index=edge_index)
        """
        A = atoms.loc[:,'x':'z'] # sample matrix
        A_dist = distance.cdist(A, A, 'euclidean') # the distance matrix
        threshold_condition = A_dist > threshold # set the element whose value is larger than threshold to 0
        A_dist[threshold_condition] = 0 # set the element whose value is larger than threshold to 0
        result = np.where(A_dist > 0)
        result = np.vstack((result[0],result[1]))
        edge_index = torch.tensor(result, dtype=torch.long)
        node_features = torch.tensor(atoms[self.features_to_use].to_numpy(), dtype=torch.float32)
        label = torch.tensor([label], dtype=torch.long)
        data = Data(x=node_features, y=label, edge_index=edge_index)
        return data

    def __compute_dist_to_center(self, data):
        """
        Given the input data matrix (n by d), return the distances of each points to the
        geometric center.
        """
        center = np.mean(data, axis=0)
        shifted_data = data - center # center the data around origin
        distances = np.sqrt(shifted_data[:,0]**2 + shifted_data[:,1]**2 + shifted_data[:,2]**2) # distances to origin
        return distances

    def __extract_sasa_data(self, siteresidue_list, pop):
        '''extracts accessible surface area data from .out file generated by POPSlegacy.
            then matches the data in the .out file to the binding site in the mol2 file.
            Used POPSlegacy https://github.com/Fraternalilab/POPSlegacy '''
        # Extracting sasa data from .out file
        residue_list = []
        qsasa_list = []
        with open(pop) as popsa:  # opening .out file
            for line in popsa:
                line_list = line.split()
                if len(line_list) == 12:  # extracting relevant information
                    residue_type = line_list[2] + line_list[4]
                    if residue_type in siteresidue_list:
                        qsasa = line_list[7]
                        residue_list.append(residue_type)
                        qsasa_list.append(qsasa)

        qsasa_list = [float(x) for x in qsasa_list]
        median = statistics.median(qsasa_list)
        qsasa_new = [median if x == '-nan' else x for x in qsasa_list]

        # Matching amino acids from .mol2 and .out files and creating dictionary
        qsasa_data = []
        fullprotein_data = list(zip(residue_list, qsasa_new))
        for i in range(len(fullprotein_data)):
            if fullprotein_data[i][0] in siteresidue_list:
                qsasa_data.append(float(fullprotein_data[i][1]))

        return qsasa_data

    def __extract_seq_entropy_data(self, siteresidue_list, profile):
        '''extracts sequence entropy data from .profile'''
        # Opening and formatting lists of the probabilities and residues
        with open(profile) as profile:  # opening .profile file
            ressingle_list = []
            probdata_list = []
            for line in profile:    # extracting relevant information
                line_list = line.split()
                residue_type = line_list[0]
                prob_data = line_list[1:]
                prob_data = list(map(float, prob_data))
                ressingle_list.append(residue_type)
                probdata_list.append(prob_data)

        ressingle_list = ressingle_list[1:]
        probdata_list = probdata_list[1:]

        # Changing single letter amino acid to triple letter with its corresponding number
        count = 0
        restriple_list = []
        for res in ressingle_list:
            newres = res.replace(res, self.__amino_single_to_triple(res))
            count += 1
            restriple_list.append(newres + str(count))

        # Calculating information entropy
        with np.errstate(divide='ignore'):      # suppress warning
            prob_array = np.asarray(probdata_list)
            log_array = np.log2(prob_array)
            log_array[~np.isfinite(log_array)] = 0  # change all infinite values to 0
            entropy_array = log_array * prob_array
            entropydata_array = np.sum(a=entropy_array, axis=1) * -1
            entropydata_list = entropydata_array.tolist()

        # Matching amino acids from .mol2 and .profile files and creating dictionary
        fullprotein_data = dict(zip(restriple_list, entropydata_list))
        seq_entropy_data = {k: float(fullprotein_data[k]) for k in siteresidue_list if k in fullprotein_data}

        return seq_entropy_data

    def __amino_single_to_triple(self, single):
        '''converts the single letter amino acid abbreviation to the triple letter abbreviation'''
        
        single_to_triple_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
                                 'G': 'GLY', 'Q': 'GLN', 'E': 'GLU', 'H': 'HIS', 'I': 'ILE',
                                 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
                                 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
        
        for i in single_to_triple_dict.keys():
            if i == single:
                triple = single_to_triple_dict[i]

        return triple


def gen_loaders(op, root_dir, pop_dir, profile_dir, training_folds, val_fold, batch_size, threshold, features_to_use, shuffle=True, num_workers=1):
    """
    Function to generate dataloaders for cross validation
    Args:
        op: operation mode, heme_vs_nucleotide, control_vs_heme or control_vs_nucleotide. 
        root_dir : folder containing all images.
        training_folds: list of integers indicating the folds, e.g: [1,2,3,4]
        val_fold: integer, which fold is used for validation, the other folds are used for training. e.g: 5
        batch_size: integer, number of data sent to GNN.
    """
    training_set = MolDatasetCV(op=op, root_dir=root_dir, pop_dir=pop_dir, profile_dir=profile_dir, folds=training_folds, threshold=threshold, features_to_use=features_to_use)
    val_set = MolDatasetCV(op=op, root_dir=root_dir, pop_dir=pop_dir, profile_dir=profile_dir, folds=[val_fold], threshold=threshold, features_to_use=features_to_use)
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, val_loader, len(training_set), len(val_set)


if __name__ == "__main__":
    pd.options.display.max_rows = 999
    args = get_args()
    op = args.op    
    root_dir = args.root_dir
    pop_dir = args.pop_dir
    profile_dir = args.profile_dir
    result_file_suffix = args.result_file_suffix
    batch_size = args.batch_size
    print('data directory:', root_dir)
    print('batch size: ', batch_size)
    num_control = args.num_control
    num_heme = args.num_heme
    num_nucleotide = args.num_nucleotide
    features_to_use = ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center']
    threshold = 4.5 # ångström

    # dataloarders
    folds = [1, 2, 3, 4, 5]
    val_fold = 5
    folds.remove(val_fold)
    train_loader, val_loader, train_size, val_size = gen_loaders(op, root_dir, pop_dir, profile_dir, folds, val_fold, batch_size=batch_size, threshold=4.5, features_to_use=features_to_use, shuffle=False, num_workers=1)

    for data in val_loader:
        print(data)
        #print('y:', data['y'])
        break