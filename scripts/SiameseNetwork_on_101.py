'''
Descripttion: 
version: 1.0
Author: ZXL
Date: 2024-05-12 13:32:15
LastEditors: ZXL
LastEditTime: 2025-10-14 17:10:29
'''
import numpy as np
import random
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool
from make_sample import SampleSetX
from nasbench.lib import model_spec as _model_spec
from get_data_from_101 import NASBench101, padding_zero_in_matrix
from data_augmentation_HAAP import create_new_metrics
from utils import train_log, get_current_time

ModelSpec = _model_spec.ModelSpec
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SiameseNetwork(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(SiameseNetwork, self).__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc_cls1 = nn.Linear(hidden_channels*2, hidden_channels*4)
        self.fc_cls2 = nn.Linear(hidden_channels*4, hidden_channels*2)
        self.fc_cls3 = nn.Linear(hidden_channels*2, output_channels)
        # self.fc_cls4 = nn.Linear(hidden_channels*2, hidden_channels)
        # self.fc_cls5 = nn.Linear(hidden_channels, output_channels)


    def forward_once(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)

        return x
    
    
    def forward(self, data1, data2):
        x1 = self.forward_once(data1)
        x2 = self.forward_once(data2)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc_cls1(x)
        x = F.relu(x)
        x = self.fc_cls2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        output = self.fc_cls3(x)
        # x = F.relu(x)
        # x = self.fc_cls4(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        # output = self.fc_cls5(x)

        return output
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
    

class SiameseNetworkDataset(Dataset):
    def __init__(self, data_sampled, tiers_num, augmentation=True):
        super(SiameseNetworkDataset, self).__init__()
        self.data_sampled = data_sampled
        self.tiers_num = tiers_num
        if augmentation:
            self.data_list = self.data_augmentation()
        else:
            self.data_list = self.data_without_augmentation()
        random.shuffle(self.data_list)
    

    def len(self):
        return len(self.data_list)
    

    def get(self, item):
        return self.data_list[item]
    

    def data_augmentation(self):
        data_sampled_aug = []
        archsize = len(self.data_sampled)
        for i in range(archsize):
            sampled_arch = self.data_sampled[i]
            arch = sampled_arch['arch']
            model_spec = ModelSpec(arch['matrix'], arch['op_list'])
            pruned_matrix, pruned_op_list = model_spec.matrix, model_spec.ops
            padding_matrix, padding_op_list = padding_zero_in_matrix(pruned_matrix, pruned_op_list)
            all_possible = create_new_metrics(padding_matrix, padding_op_list[1:-1])
            for j in range(len(all_possible)):
                new_sampled_arch = copy.deepcopy(sampled_arch)
                op_list = []
                op_list.append('input')
                op_list.extend(all_possible[j]['module_integers'])
                op_list.append('output')
                arch['matrix'], arch['op_list'] = all_possible[j]['module_adjacency'], op_list
                new_sampled_arch['arch'] = arch
                data_sampled_aug.append(new_sampled_arch)
        archsize_aug = len(data_sampled_aug)
        pair_index = torch.triu_indices(archsize_aug, archsize_aug, offset=1)
        pair_num = pair_index.shape[1]
        oper2feature = {'input': [1, 0, 0, 0, 0, 0], 'conv1x1-bn-relu': [0, 1, 0, 0, 0, 0], 'conv3x3-bn-relu': [0, 0, 1, 0, 0, 0], 
                        'maxpool3x3': [0, 0, 0, 1, 0, 0], 'null': [0, 0, 0, 0, 1, 0], 'output':[0, 0, 0, 0, 0, 1]}
        data_aug = []
        for index in range(pair_num):
            pair1_index = pair_index[0][index]
            pair2_index = pair_index[1][index]
            pair1 = data_sampled_aug[pair1_index]
            pair2 = data_sampled_aug[pair2_index]
            arch1 = pair1['arch']
            arch2 = pair2['arch']
            x1 = [oper2feature[oper] for oper in arch1['op_list']]
            x2 = [oper2feature[oper] for oper in arch2['op_list']]
            indices1 = np.where(arch1['matrix'] == 1)
            indices1 = np.array(indices1)
            indices2 = np.where(arch2['matrix'] == 1)
            indices2 = np.array(indices2)
            edge_index1 = torch.tensor(indices1, dtype=torch.long)
            edge_index2 = torch.tensor(indices2, dtype=torch.long)
            if pair1['acc'] > pair2['acc']:
                lable = 1
            else:
                lable = 0
            data1 = Data(x=x1, edge_index=edge_index1)
            data2 = Data(x=x2, edge_index=edge_index2)
            input = (data1, data2)
            data = (input, lable)
            data_aug.append(data)
        
        return data_aug
    

    def data_without_augmentation(self):
        data_sampled_without = []
        archsize = len(self.data_sampled)
        for i in range(archsize):
            sampled_arch = self.data_sampled[i]
            arch = sampled_arch['arch']
            new_sampled_arch = copy.deepcopy(sampled_arch)
            # pruned
            model_spec = ModelSpec(arch['matrix'], arch['op_list'])
            pruned_matrix, pruned_op_list = model_spec.matrix, model_spec.ops
            padding_matrix, padding_op_list = padding_zero_in_matrix(pruned_matrix, pruned_op_list)
            arch['matrix'], arch['op_list'] = padding_matrix, padding_op_list
            new_sampled_arch['arch'] = arch
            data_sampled_without.append(new_sampled_arch)
        tier_size = int(archsize / 2)
        sorted_data_sampled_without = sorted(data_sampled_without, key=lambda x:x['acc'], reverse=True)
        data_sampled = sorted_data_sampled_without[0:tier_size]
        random.shuffle(data_sampled)
        pair_index = torch.triu_indices(tier_size, tier_size, offset=1)
        pair_num = pair_index.shape[1]
        oper2feature = {'input': [1, 0, 0, 0, 0, 0], 'conv1x1-bn-relu': [0, 1, 0, 0, 0, 0], 'conv3x3-bn-relu': [0, 0, 1, 0, 0, 0], 
                        'maxpool3x3': [0, 0, 0, 1, 0, 0], 'null': [0, 0, 0, 0, 1, 0], 'output':[0, 0, 0, 0, 0, 1]}
        data_without = []
        z1 = 0
        z2 = 0
        for index in range(pair_num):
            pair1_index = pair_index[0][index]
            pair2_index = pair_index[1][index]
            pair1 = data_sampled[pair1_index]
            pair2 = data_sampled[pair2_index]
            arch1 = pair1['arch']
            arch2 = pair2['arch']
            x1 = [oper2feature[oper] for oper in arch1['op_list']]
            x2 = [oper2feature[oper] for oper in arch2['op_list']]
            x1 = torch.tensor(x1, dtype=torch.float)
            x2 = torch.tensor(x2, dtype=torch.float)
            indices1 = np.where(arch1['matrix'] == 1)
            indices1 = np.array(indices1)
            indices2 = np.where(arch2['matrix'] == 1)
            indices2 = np.array(indices2)
            edge_index1 = torch.tensor(indices1, dtype=torch.long)
            edge_index2 = torch.tensor(indices2, dtype=torch.long)
            if pair1['acc'] > pair2['acc']:
                lable = 1
                z1 += 1
            else:
                lable = 0
                z2 += 1
            data1 = Data(x=x1, edge_index=edge_index1)
            data2 = Data(x=x2, edge_index=edge_index2)
            input = (data1, data2)
            data = (input, lable)
            data_without.append(data)
        
        print(z1,z2)
        
        return data_without
    

def train_SiameseNetwork(model, EPOCH, train_loader, optimizer, criterion):
    model.train()
    for epoch in range(EPOCH):
        correct = 0
        running_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            ((data1, data2), label) = data
            data1, data2, label = data1.to(device), data2.to(device), label.to(device)
            output = model(data1, data2)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
        loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / len(train_loader.dataset)
        info = 'Epoch :  [{:03d}/{}]'.format(epoch+1, EPOCH) + '   |   loss :  {:08f}'.format(loss) + '   |   train_accuracy :  {:08f}%'.format(train_accuracy)
        print(info)

    return model


def train_SiameseNetwork_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    correct = 0
    running_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        ((data1, data2), label) = data
        data1, data2, label = data1.to(device), data2.to(device), label.to(device)
        output = model(data1, data2)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
        _, predicted = torch.max(output, 1)
        correct += (predicted == label).sum().item()
    loss = running_loss / len(train_loader)
    accuracy = 100 * correct / len(train_loader.dataset)

    return loss, accuracy


def test_SiameseNetwork(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        for data in test_loader:
            ((data1, data2), label) = data
            data1, data2, label = data1.to(device), data2.to(device), label.to(device)
            output = model(data1, data2)
            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
    accuracy = 100 * correct / len(test_loader.dataset)
    
    return accuracy




if __name__ == '__main__':
    # setting
    EPOCH = 300
    batch_size = 64
    tiers_num = 5
    threshold = 0.5

    save_path = r'./pkl/sampleset_200.pkl'
    with open(save_path, 'rb') as file:
        sampleset = pickle.load(file)
    data_sampled = sampleset.get_data_sampled()
    dataset = SiameseNetworkDataset(data_sampled=data_sampled, tiers_num=tiers_num, augmentation=False)

    num_split = 0.7
    num_train = int(len(dataset) * num_split)
    train_dataset = dataset[:num_train]
    test_dataset = dataset[num_train:]

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    SNet = SiameseNetwork(input_channels=6, hidden_channels=64, output_channels=2).to(device)
    print(SNet)
    optimizer = optim.Adam(SNet.parameters())
    criterion = nn.CrossEntropyLoss()

    # train_SiameseNetwork(TNet, EPOCH, train_loader, optimizer, criterion)
    for epoch in range(EPOCH):
        loss, train_accuracy = train_SiameseNetwork_one_epoch(SNet, train_loader, optimizer, criterion)
        test_accuracy = test_SiameseNetwork(SNet, test_loader)
        info1 = 'Epoch :  [{:03d}/{}]'.format(epoch+1, EPOCH) + '   |   loss :  {:08f}'.format(loss)
        info2 = '   |   train_accuracy :  {:08f}%'.format(train_accuracy) + '   |  test_accuracy :  {:08f}%'.format(test_accuracy)
        info = info1 + info2
        print(info)
    suffix_num = get_current_time()
    save_path = r'./predictor/Queries Number/{}_SiameseNetwork_{}.pt'.format(200, suffix_num)
    torch.save(SNet.state_dict(), save_path)
    