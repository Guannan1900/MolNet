"""
Graph classification with Graph Neural Networks (GNNs). 
Part of the code is adapted from this example:
https://github.com/rusty1s/pytorch_geometric/blob/master/examples/mutag_gin.py

"""
import argparse
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from dataloader import gen_loaders
from torch_geometric.nn import SplineConv, global_add_pool
import matplotlib.pyplot as plt
import numpy as np


def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-op',
                        default='control_vs_nucleotide',
                        required=False,
                        choices = ['control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'],
                        help="'control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'")
    parser.add_argument('-root_dir',
                        default='../MolNet-data/',
                        required=False,
                        help='directory to load data for 5-fold cross-validation.')                         
    parser.add_argument('-num_control',
                        default=1946,
                        required=False,
                        help='number of control data points, used to calculate the positive weight for the loss function')
    parser.add_argument('-num_heme',
                        default=596,
                        required=False,
                        help='number of heme data points, used to calculate the positive weight for the loss function.')
    parser.add_argument('-num_nucleotide',
                        default=1553,
                        required=False,
                        help='number of num_nucleotide data points, used to calculate the positive weight for the loss function.')
    return parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dim, d):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim, kernel_size=5, aggr='add')
        self.conv2 = SplineConv(32, 64, dim, kernel_size=5, aggr='add')
        self.conv3 = SplineConv(64, 64, dim, kernel_size=5, aggr='add')
        self.conv4 = SplineConv(64, 64, dim, kernel_size=5, aggr='add')
        #self.conv5 = SplineConv(64, 64, dim=3, kernel_size=5, aggr='add')
        #self.conv6 = SplineConv(64, 64, dim=3, kernel_size=5, aggr='add')
        self.lin1 = torch.nn.Linear(64, 256)
        self.lin2 = torch.nn.Linear(256, d.num_nodes)

    def forward(self, data):
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, pseudo))
        x = F.elu(self.conv2(x, edge_index, pseudo))
        x = F.elu(self.conv3(x, edge_index, pseudo))
        x = F.elu(self.conv4(x, edge_index, pseudo))
        #x = F.elu(self.conv5(x, edge_index, pseudo))
        #x = F.elu(self.conv6(x, edge_index, pseudo))
        x = F.elu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


def train(epoch):
    model.train()

    if epoch == 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
    
    correct = 0

    for data in train_loader:
        optimizer.zero_grad()
        train_loss = F.nll_loss(model(data.to(device)), target)
        train_loss.backward()
        optimizer.step()

        pred = model(data.to(device)).max(1)[1]
        correct += pred.eq(target).sum().item()
        acc = correct / (train_size * d.num_nodes)
    return train_loss, acc

def validate():
    """
    Returns loss and accuracy on validation set.
    Global vars: val_loader, val_size, device, model
    """
    model.eval()
    correct = 0

    for data in val_loader:
        val_loss = F.nll_loss(model(data.to(device)), target)
        pred = model(data.to(device)).max(1)[1]
        correct += pred.eq(target).sum().item()
    acc = correct / (val_size * d.num_nodes)
    return val_loss, acc
    

def plot_loss(train_loss, val_loss, loss_dir, num_epoch):
    """
    Plot loss.
    """
    epochs = np.array(range(num_epoch), dtype=int) + 1
    fig = plt.figure()
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')    
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    plt.plot(epochs, train_loss, color='b', label='training loss')
    plt.plot(epochs, val_loss, color='r', label='validation loss')
    plt.legend()
    plt.savefig(loss_dir)


def plot_accuracy(train_acc, val_acc, acc_dir, num_epoch):
    """
    Plot accuracy.
    """
    epochs = np.array(range(num_epoch), dtype=int) + 1
    fig = plt.figure()
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')    
    train_acc = np.array(train_acc)
    val_acc = np.array(val_acc)
    plt.plot(epochs, train_acc, color='b', label='training accuracy')
    plt.plot(epochs, val_acc, color='r', label='validation accuracy')
    plt.legend()
    plt.savefig(acc_dir)


if __name__ == "__main__":
    torch.manual_seed(42)
    args = get_args()
    op = args.op    
    root_dir = args.root_dir
    print('data directory:', root_dir)
    num_control = args.num_control
    num_heme = args.num_heme
    num_nucleotide = args.num_nucleotide
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # detect cpu or gpu

    threshold = 4.5 # unit:ångström. hyper-parameter for forming graph, distance thresh hold of forming edge.
    num_epoch = 100 # number of epochs to train
    batch_size = 4
    num_workers = 4 # number of processes assigned to dataloader.
    neural_network_size = 4
    
    print('threshold:', threshold)    
    print('number of epochs:', num_epoch)
    print('batch_size',batch_size)
    print('number of data loader workers:', num_workers)
    print('neural network size:', neural_network_size)

    # dataloarders
    folds = [1, 2, 3, 4, 5]
    val_fold = 1
    folds.remove(val_fold)
    train_loader, val_loader, train_size, val_size = gen_loaders(op, root_dir, folds, val_fold, batch_size=batch_size, threshold=threshold, shuffle=True, num_workers=num_workers)
    model = Net(dim=neural_network_size).to(device)
    print('model architecture:')
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001, amsgrad=False)
    print('optimizer:')
    print(optimizer)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = 9999999
    print('begin training...')
    for epoch in range(1, 1 + num_epoch):
        train_loss, train_acc = train(epoch)
        val_loss, val_acc = validate()
        print('Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, Val Loss: {:.7f}, Val Acc: {:.7f}'.format(epoch, train_loss, train_acc, val_loss, val_acc))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_dict = {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss':val_loss, 'val_acc':val_acc}
    print('results at minimum val loss:')
    print(best_val_loss_dict)
    plot_loss(train_losses, val_losses, './figure/gin_loss_6.png', num_epoch)  
    plot_accuracy(train_accs, val_accs, './figure/gin_acc_6.png', num_epoch)  

    '''
    # 5-fold cross-validation
    for i in range(5):
        print('*********************************************************************')
        print('starting {}th fold cross-validation'.format(i+1))
        
        # dataloarders
        folds = [1, 2, 3, 4, 5]
        val_fold = i+1
        folds.remove(val_fold)
        train_loader, val_loader, train_size, val_size = gen_loaders(op, root_dir, folds, val_fold, batch_size=batch_size, threshold=threshold, shuffle=True, num_workers=num_workers)

        model = Net(num_features=3, dim=neural_network_size).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, 1 + num_epoch):
            train_loss, train_acc = train(epoch)
            val_loss, val_acc = validate()
            print('Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, Val Loss: {:.7f}, Val Acc: {:.7f}'.format(epoch, train_loss, train_acc, val_loss, val_acc))
    '''

    '''
    TO DO:
    1. Save the results of 5 folds when validation loss is at minimum.
    2. Compute avg accuracy across 5 folds.
    '''