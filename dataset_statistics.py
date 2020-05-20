import argparse
import networkx as nx
from dataloader import gen_loaders
import numpy as np
from torch_geometric.utils import to_networkx
from networkx.classes.function import info
import statistics 
from tqdm import tqdm

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
    return parser.parse_args()


def compute_graph_stat(graph):
    """
    Compute the following statistics of graph:
        1. number of nodes.
        2. number of edges.
        4. number of isolated nodes.
        5. diameter of graph.
        6. density of graph.
    """
    graph = to_networkx(graph)
    graph = graph.to_undirected()
    graph_stat = {}
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    dens = nx.classes.function.density(graph)
    try:
        dia = nx.algorithms.distance_measures.diameter(graph) # can be inf when graph is not connected
    except:
        dia = float('inf')
    graph_stat['num_nodes'] = num_nodes
    graph_stat['num_edges'] = num_edges
    graph_stat['density'] = dens
    graph_stat['diameter'] = dia

    degrees = list(graph.degree())
    degrees = [x[1] for x in degrees]
    sum_of_edges = sum(degrees)
    avg_degree = sum_of_edges/num_nodes
    graph_stat['avg_degree'] = avg_degree

    #print('average degree:', avg_degree)
    #print(info(graph))
    #print('density of graph:', dens)
    #print('diameter of graph:', dia)
    return graph_stat


if __name__=="__main__":
    args = get_args()
    op = args.op    
    root_dir = args.root_dir
    threshold = 4.5
    folds = [1, 2, 3, 4, 5]
    val_fold = 5
    folds.remove(val_fold)
    train_loader, val_loader, train_size, val_size = gen_loaders(op, root_dir, folds, val_fold, batch_size=1, threshold=threshold, shuffle=False, num_workers=1)
    print('length of training set:', train_size)
    print('length of validation set:', val_size)

    num_nodes = []
    num_edges = []
    density = []
    diameter = []
    avg_degree = []
    for graph in tqdm(train_loader):
        graph_stat = compute_graph_stat(graph)
        num_nodes.append(graph_stat['num_nodes'])
        num_edges.append(graph_stat['num_edges'])
        density.append(graph_stat['density'])
        diameter.append(graph_stat['diameter'])
        avg_degree.append(graph_stat['avg_degree'])
        
    for graph in tqdm(val_loader):
        graph_stat = compute_graph_stat(graph)
        num_nodes.append(graph_stat['num_nodes'])
        num_edges.append(graph_stat['num_edges'])
        density.append(graph_stat['density'])
        diameter.append(graph_stat['diameter'])
        avg_degree.append(graph_stat['avg_degree'])

    
    num_nodes_mean = statistics.mean(num_nodes) 
    num_edges_mean = statistics.mean(num_edges)
    density_mean = statistics.mean(density) 
    diameter_mean = statistics.mean(diameter)
    avg_degree_mean = statistics.mean(avg_degree)

    num_nodes_median = statistics.median(num_nodes) 
    num_edges_median = statistics.median(num_edges)
    density_median = statistics.median(density) 
    diameter_median = statistics.median(diameter)
    avg_degree_median = statistics.median(avg_degree)

    print('dataset:', op)
    print('mean number of nodes:', num_nodes_mean)
    print('mean number of edges:', num_edges_mean)
    print('mean density:', density_mean)
    print('mean diameter:', diameter_mean)
    print('mean average degree:', avg_degree_mean)
    print('median number of nodes:', num_nodes_median)
    print('median number of edges:', num_edges_median)
    print('median density:', density_median)
    print('median diameter:', diameter_median)
    print('median average degree:', avg_degree_median)
    

