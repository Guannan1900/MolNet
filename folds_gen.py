"""
Generate files containing the 5 pre-divided folds of the data. 
"""
import os

if __name__ == "__main__":
    root_dir = '../MolNet-data/pockets/' # directory of the pocket data
    pocket_classes = ['control', 'nucleotide', 'heme']
    folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
    for pocket_class in pocket_classes:
        for fold in folds:
            pocket_dir =  root_dir + pocket_class + '/' + fold + '/' # read file names from
            pocket_list = [name[:-9] for name in os.listdir(pocket_dir) if os.path.isfile(os.path.join(pocket_dir, name))]
            out_dir = './folds/' + pocket_class + '_' + fold + '.txt' # output file names to

            print(pocket_list)
            print(pocket_dir)
            print(out_dir)
            print('---------------------') 
            
            f = open(out_dir, 'w') # overwrite
            for pocket in pocket_list:
                f.write(pocket + '\n')
           
            