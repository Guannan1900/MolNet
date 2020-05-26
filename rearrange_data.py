import argparse
from os import listdir
from os.path import isfile, join
import os
import shutil

def get_args():
    parser = argparse.ArgumentParser('python')

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
    return parser.parse_args()

def create_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        os.makedirs(dir)

if __name__ == "__main__":
    args = get_args()
    root_dir = args.root_dir
    pop_dir = args.pop_dir
    profile_dir = args.profile_dir
    pocket_classes = ['control/', 'nucleotide/', 'heme/']
    folds = ['fold_1/', 'fold_2/', 'fold_3/', 'fold_4/', 'fold_5/']

    for pocket_class in pocket_classes:
        for fold in folds:
            create_dir(pop_dir + pocket_class + fold)
            create_dir(profile_dir + pocket_class + fold)

            mol_dir = root_dir + pocket_class + fold
            mol_files = [f for f in listdir(mol_dir) if isfile(join(mol_dir, f))]
            pockets = [x[:-9] for x in mol_files]
            pops = [x + '.out' for x in pockets]
            profiles = [x + '.profile' for x in pockets]

            for pop in pops:
                shutil.move(pop_dir + pocket_class + pop, pop_dir + pocket_class + fold + pop)
            
            for profile in profiles:
                shutil.move(profile_dir + pocket_class + profile, profile_dir + pocket_class + fold + profile)
        

            #print(mol_dir)
            #print(mol_files)
            #print(pockets)
            #print(pops)
            #print(profiles)