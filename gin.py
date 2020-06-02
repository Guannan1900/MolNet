"""
Graph classification with Graph Neural Networks (GNNs). 
Part of the code is adopted from this example:
https://github.com/rusty1s/pytorch_geometric/blob/master/examples/mutag_gin.py

"""
import argparse
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LeakyReLU
from dataloader import gen_loaders
from torch_geometric.nn import GINConv, global_add_pool
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import json


def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-op',
                        default='control_vs_nucleotide',
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
    parser.add_argument('-roc_path',
                        default='./roc/gin_5fold_7.json',
                        required=False,
                        help='path of roc data of 5 folds.')                                                 
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
        nn1 = Sequential(Linear(num_features, dim), LeakyReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        #nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        #self.conv5 = GINConv(nn5)
        #self.bn5 = torch.nn.BatchNorm1d(dim)

        #nn6 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        #self.conv6 = GINConv(nn6)
        #self.bn6 = torch.nn.BatchNorm1d(dim)

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
        #x = F.relu(self.conv5(x, edge_index))
        #x = self.bn5(x)
        #x = F.relu(self.conv6(x, edge_index))
        #x = self.bn6(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def train(epoch, lr_decay_epoch):
    """
    Train the model for 1 epoch, then return the averaged loss of the data 
    in this epoch.
    Global vars: train_loader, train_size, device, optimizer, model
    """
    model.train()
    if epoch == lr_decay_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_total = 0
    epoch_pred = [] # all the predictions for the epoch
    epoch_label = [] # all the labels for the epoch
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_total += loss.item() * data.num_graphs
        optimizer.step()
        pred = output.max(dim=1)[1]
        output_prob = torch.exp(output) # output probabilities of each class
    
        pred_cpu = list(pred.cpu().detach().numpy()) # used to compute evaluation metrics
        label = list(data.y.cpu().detach().numpy()) # used to compute evaluation metrics
        
        epoch_pred.extend(pred_cpu)
        epoch_label.extend(label)

    acc, precision, recall, f1, mcc = compute_metrics(epoch_label, epoch_pred)
    train_loss = loss_total / train_size # averaged training loss
    result_dict = {'acc':acc, 'precision': precision, 'recall': recall, 'f1':f1, 'mcc': mcc, 'loss': train_loss}   
    return result_dict


def validate():
    """
    Returns loss and accuracy on validation set.
    Global vars: val_loader, val_size, device, model
    """
    model.eval()

    loss_total = 0
    epoch_pred = [] # all the predictions for the epoch
    epoch_label = [] # all the labels for the epoch
    epoch_prob = [] # all the output probabilities
    for data in val_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss_total += loss.item() * data.num_graphs
        pred = output.max(dim=1)[1]
        output_prob = torch.exp(output) # output probabilities of each class
        
        pred_cpu = list(pred.cpu().detach().numpy()) # used to compute evaluation metrics
        label = list(data.y.cpu().detach().numpy()) # used to compute evaluation metrics
        output_prob_cpu = output_prob.cpu().detach().numpy() # softmax output
        output_prob_cpu = list(output_prob_cpu[:,1]) # probability of positive class

        epoch_pred.extend(pred_cpu)
        epoch_label.extend(label)
        epoch_prob.extend(output_prob_cpu)
        
    val_loss = loss_total / val_size # averaged training loss
    acc, precision, recall, f1, mcc = compute_metrics(epoch_label, epoch_pred) # evaluation metrics
    result_dict = {'acc':acc, 'precision': precision, 'recall': recall, 'f1':f1, 'mcc': mcc, 'loss': val_loss}   
    roc_dict = {'label':epoch_label, 'prob': epoch_prob}    # data needed to compute roc curve 
    return result_dict, roc_dict


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


def compute_metrics(label, out):
    """
    Compute the evaluation metrics of the model.
    Both label and out should be converted from Pytorch tensor to numpy arrays containing 0s and 1s.
    """
    acc = metrics.accuracy_score(label, out)
    precision = metrics.precision_score(label,out)
    recall = metrics.recall_score(label,out)
    f1 = metrics.f1_score(label,out)
    mcc = metrics.matthews_corrcoef(label, out)
    return acc, precision, recall, f1, mcc


if __name__ == "__main__":
    torch.manual_seed(42)
    args = get_args()
    op = args.op    
    root_dir = args.root_dir
    pop_dir = args.pop_dir
    profile_dir = args.profile_dir
    roc_path = args.roc_path
    print('data directory:', root_dir)
    print('pop directory (for sasa feature):', pop_dir)
    print('profile directory (for sequence_entropy feature):', profile_dir)
    num_control = args.num_control
    num_heme = args.num_heme
    num_nucleotide = args.num_nucleotide
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # detect cpu or gpu

    threshold = 4.5 # unit: ångström. hyper-parameter for forming graph, distance thresh hold of forming edge.
    num_epoch = 2 # number of epochs to train
    lr_decay_epoch = 800
    batch_size = 4
    num_workers = batch_size # number of processes assigned to dataloader.
    neural_network_size = 16
    # Should be subset of ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sasa', 'sequence_entropy']
    #features_to_use = ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sasa', 'sequence_entropy']
    features_to_use = ['hydrophobicity']

    num_features = len(features_to_use)

    print('threshold:', threshold)    
    print('number of epochs:', num_epoch)
    print('learning rate decay at epoch:', lr_decay_epoch)
    print('batch_size:',batch_size)
    print('number of data loader workers:', num_workers)
    print('neural network size:', neural_network_size)
    print('features to use:', features_to_use)
    
    results = []
    rocs = []
    for val_fold in [1,2,3,4,5]:
        folds = [1, 2, 3, 4, 5]
        folds.remove(val_fold)
        train_loader, val_loader, train_size, val_size = gen_loaders(op, root_dir, pop_dir, profile_dir, folds, val_fold, batch_size=batch_size, threshold=threshold, features_to_use=features_to_use, shuffle=True, num_workers=num_workers)
        model = Net(num_features=num_features, dim=neural_network_size).to(device)
        print('model architecture:')
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001, amsgrad=False)
        print('optimizer:')
        print(optimizer)
        #train_losses = []
        #val_losses = []
        #train_accs = []
        #val_accs = []
        best_val_loss = 9999999
        print('begin training...')
        for epoch in range(1, 1 + num_epoch):
            train_results = train(epoch, lr_decay_epoch)
            val_results, roc_dict = validate()
            print('epoch: ', epoch)
            print('train: ', train_results)
            print('validation: ', val_results)            
            #train_losses.append(train_results['loss'])
            #val_losses.append(val_results['loss'])
            #train_accs.append(train_results['acc'])
            #val_accs.append(val_results['acc'])
            if val_results['loss'] < best_val_loss:
                best_val_loss_dict = {}
                best_val_loss = val_results['loss']
                best_val_loss_dict['epoch'] = epoch
                best_val_loss_dict['train'] = train_results
                best_val_loss_dict['val'] = val_results

                fpr, tpr, thresholds = metrics.roc_curve(roc_dict['label'], roc_dict['prob'])
                fpr = fpr.tolist()
                tpr = tpr.tolist()
                thresholds = thresholds.tolist()
                best_val_loss_roc = {'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds}
        print('results at minimum val loss:')
        print(best_val_loss_dict)
        results.append(best_val_loss_dict)
        rocs.append(best_val_loss_roc)
        #plot_loss(train_losses, val_losses, './figure/gin_loss_6.png', num_epoch)  
        #plot_accuracy(train_accs, val_accs, './figure/gin_acc_6.png', num_epoch)  
    
    # averaged results over folds
    print('*****************************************************************')
    train_loss = 0
    train_acc = 0
    train_precision = 0
    train_recall = 0
    train_f1 = 0
    train_mcc =0

    val_loss = 0
    val_acc = 0
    val_precision = 0
    val_recall = 0
    val_f1 = 0
    val_mcc =0

    best_val_loss_epochs = []
    for best_val_loss_dict in results:
        best_val_loss_epochs.append(best_val_loss_dict['epoch'])

        train_loss += best_val_loss_dict['train']['loss']
        train_acc += best_val_loss_dict['train']['acc']
        train_precision += best_val_loss_dict['train']['precision']
        train_recall += best_val_loss_dict['train']['recall']        
        train_f1 += best_val_loss_dict['train']['f1']
        train_mcc += best_val_loss_dict['train']['mcc']    

        val_loss += best_val_loss_dict['val']['loss']
        val_acc += best_val_loss_dict['val']['acc']
        val_precision += best_val_loss_dict['val']['precision']
        val_recall += best_val_loss_dict['val']['recall']        
        val_f1 += best_val_loss_dict['val']['f1']
        val_mcc += best_val_loss_dict['val']['mcc']

    train_loss = train_loss/5
    train_acc = train_acc/5
    train_precision = train_precision/5
    train_recall = train_recall/5
    train_f1 = train_f1/5
    train_mcc = train_mcc/5

    val_loss = val_loss/5
    val_acc = val_acc/5
    val_precision = val_precision/5
    val_recall = val_recall/5
    val_f1 = val_f1/5
    val_mcc = val_mcc/5
    
    print('averaged performance at best validation loss:')
    print('epochs that have best validation loss:', best_val_loss_epochs)

    print('averaged train loss:', train_loss)
    print('averaged train accuracy:', train_acc)
    print('averaged train precision:', train_precision)
    print('averaged train recall:', train_recall)
    print('averaged train f1:', train_f1)
    print('averaged train mcc:', train_mcc)

    print('averaged val loss:', val_loss)
    print('averaged val accuracy:', val_acc)
    print('averaged val precision:', val_precision)
    print('averaged val recall:', val_recall)
    print('averaged val f1:', val_f1)
    print('averaged val mcc:', val_mcc)

    with open(roc_path, 'w') as fp:
        json.dump(rocs, fp) # save roc files as a json file
    
