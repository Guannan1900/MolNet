"""
Graph classification with Graph Neural Networks (GNNs). 
Part of the code is adapted from this example:
https://github.com/rusty1s/pytorch_geometric/blob/master/examples/mutag_gin.py

"""
import argparse
import torch
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
    parser.add_argument('-result_file_suffix',
                        required=True,
                        help='suffix to result file')                        
    parser.add_argument('-batch_size',
                        type=int,
                        default=32,
                        required=False,
                        help='the batch size, normally 2^n.')
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
    def __init__(self):
        super(Net, self).__init__()

        num_features = dataset.num_features
        dim = 32

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
        self.fc2 = Linear(dim, dataset.num_classes)

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
    """
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_total = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_total += loss.item() * data.num_graphs
        optimizer.step()
    return loss_total / len(train_dataset)
    

if __name__ == "__main__":
    torch.manual_seed(42)
    args = get_args()
    op = args.op    
    root_dir = args.root_dir
    result_file_suffix = args.result_file_suffix
    batch_size = args.batch_size
    print('data directory:', root_dir)
    print('batch size: '+str(batch_size))
    num_control = args.num_control
    num_heme = args.num_heme
    num_nucleotide = args.num_nucleotide
    
    # cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 5-fold cross-validation
    for i in range(5):
        print('*********************************************************************')
        print('starting {}th fold cross-validation'.format(i+1))
        
        # dataloarders
        folds = [1, 2, 3, 4, 5]
        val_fold = i+1
        folds.remove(val_fold)
        train_loader, val_loader = gen_loaders(op, root_dir, folds, val_fold, batch_size, shuffle=True, num_workers=1)

        # model
        model = Net().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


        break