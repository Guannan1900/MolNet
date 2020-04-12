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
from torch_geometric.nn import GINConv, global_add_pool


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
    def __init__(self, num_features, dim):
        super(Net, self).__init__()
        '''
        num_features = dataset.num_features
        dim = 32
        '''
        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, 2) # binary classification, softmax is used instead of sigmoid here.

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def train(epoch):
    """
    Train the model for 1 epoch, then return the averaged loss of the data 
    in this epoch.
    Global vars: train_loader, train_size, device, optimizer, model
    """
    model.train()
    '''
    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']
    '''
    loss_total = 0
    correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_total += loss.item() * data.num_graphs
        optimizer.step()
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    train_loss = loss_total / train_size # averaged training loss
    acc = correct / train_size # accuracy        
    return train_loss, acc


def validate():
    """
    Returns loss and accuracy on validation set.
    Global vars: val_loader, val_size, device, model
    """
    model.eval()

    loss_total = 0
    correct = 0
    for data in val_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss_total += loss.item() * data.num_graphs
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    val_loss = loss_total / val_size # averaged training loss
    acc = correct / val_size # accuracy        
    return val_loss, acc


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
    num_workers = 4 # number of processes assigned to dataloader.
    num_epoch = 5 # number of epochs to train
    batch_size = 4

    # 5-fold cross-validation
    for i in range(5):
        print('*********************************************************************')
        print('starting {}th fold cross-validation'.format(i+1))
        
        # dataloarders
        folds = [1, 2, 3, 4, 5]
        val_fold = i+1
        folds.remove(val_fold)
        train_loader, val_loader, train_size, val_size = gen_loaders(op, root_dir, folds, val_fold, batch_size=batch_size, threshold=threshold, shuffle=True, num_workers=num_workers)

        model = Net(num_features=3, dim=8).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, 1 + num_epoch):
            train_loss, train_acc = train(epoch)
            val_loss, val_acc = validate()
            print('Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, Val Loss: {:.7f}, Val Acc: {:.7f}'.format(epoch, train_loss, train_acc, val_loss, val_acc))

    '''
    TO DO:
    1. Save the results of 5 folds when validation loss is at minimum.
    2. Compute avg accuracy across 5 folds.
    '''
