# Mo-Net
Mo-Net is short for Molecular-Network. It is a deep learning based-frame work to classify ligand binding sites. Graphs are formed from mol2 files then classified by graph neural networks (GNNs).

## Folders
* cmaps: color map containing information for atom types and hydrophobicity.
* data: mol2 files that are pre-divided into 5 folds for cross-validation.
* figure: figures generated by the program.
* legayc: codes developed but not used in final version.

## Dataset Statistics
The following two tables list the statistics of the graph data.
1. Control against nucleotide:   

|  | number of nodes | number of edges | density | diameter | average degree |   
| --- | ---             | ---             | ---     | ---      | ---            |   
| mean | 176.31 | 971.03 | 0.071 | inf | 10.89 |   
| median | 168.0 | 922.0 | 0.066 | 10 | 10.93 |   

2. Control against heme:   

|  | number of nodes | number of edges | density | diameter | average degree |
| --- | --- | --- | --- | --- | --- |
| mean | 161.23 | 878.91 | 0.074 | inf | 10.81 |
| median | 157 | 844 | 0.070 | 10 | 10.83 |

Note that the mean diameter is inf because there are graphs that are not connected.

## Experiment results
Talbles below show the performance (validation) of GNNs trained on the datasets pre-divided into 5 folds. 
1. Control against nucleotide:   

| model | accuracy | precision | recall | F1 | MCC | AUC | log |   
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: | :---: |    
| GIN | 0.860 | 0.819 | 0.879 | 0.848 | 0.720 | 0.935 | gin_5fold_control_atp_1.txt |   
   
2. Control against heme:   

| model | accuracy | precision | recall | F1 | MCC | AUC | log |
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: | :---: |   
| GIN |          |           |        |    |     |     |     |

