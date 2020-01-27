import argparse
import numpy as np 
import pandas as pd 
from biopandas.mol2 import PandasMol2
import networkx as nx 
from torch.utils.data import Dataset, DataLoader

"""
Questions before starting: 
 1. How to represent graph data structure? NetworkX or Pytorch geometric data?
    - use NextworkX first then convert to Pytorch geometric data format.
 2. Pandas or BioPandas?
    - BioPandas
 3. Batch?
 4. Normalize features?
"""


class MolDatasetCV(Dataset):
    """
    Dataset for MolNet, can be used to load multiple folds for training or single fold for validation and testing
    """
    def __init__(self, op, root_dir, folds, transform=None):
        """
        Args:
            op: operation mode, heme_vs_nucleotide, control_vs_heme or control_vs_nucleotide. 
            root_dir: folder containing all images.
            folds: a list containing folds to generate training data, for example [1,2,3,4], [5].
            transform: transform to be applied to images.
        """
        self.op = op
        self.root_dir = root_dir
        self.folds = folds
        self.transform = transform
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
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        # get image directory
        folder_idx, sub_idx = self.__locate_file(idx)
        img_dir = self.list_of_files_list[folder_idx][sub_idx]
        
        # read image
        image = io.imread(img_dir)

        # get label 
        label = self.__get_class_int(folder_idx)

        # apply transform to PIL image if applicable
        if self.transform:
            image = self.transform(image)

        return image, label

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


def read_mol(mol_path):
    """
    Read the mol2 file as a dataframe.
    """
    mol2 = PandasMol2().read_mol2(bs)
    atoms = mol2.df[['atom_id','subst_name', 'atom_type', 'atom_name', 'x', 'y', 'z', 'charge']]


if __name__ == "__main__":