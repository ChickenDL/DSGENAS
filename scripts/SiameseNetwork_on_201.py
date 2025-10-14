'''
Descripttion: 
version: 1.0
Author: ZXL
Date: 2024-05-12 13:32:15
LastEditors: ZXL
LastEditTime: 2025-10-14 17:11:24
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
        data_aug = []
        
        return data_aug
    

    def data_without_augmentation(self):
        data_sampled_without = []
        archsize = len(self.data_sampled)
        for i in range(archsize):
            sampled_arch = self.data_sampled[i]
            new_sampled_arch = copy.deepcopy(sampled_arch)
            data_sampled_without.append(new_sampled_arch)
        tier_size = int(archsize / 2)
        sorted_data_sampled_without = sorted(data_sampled_without, key=lambda x:x['acc'], reverse=True)
        data_sampled = sorted_data_sampled_without[0:tier_size]
        random.shuffle(data_sampled)
        pair_index = torch.triu_indices(tier_size, tier_size, offset=1)
        pair_num = pair_index.shape[1]
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
            matrix1, op_list1 = arch1['matrix'], arch1['op_list']
            matrix2, op_list2 = arch2['matrix'], arch2['op_list']
            data1 = arch2data(matrix1, op_list1)
            data2 = arch2data(matrix2, op_list2)
            if pair1['acc'] > pair2['acc']:
                lable = 1
                z1 += 1
            else:
                lable = 0
                z2 += 1
            input = (data1, data2)
            data = (input, lable)
            data_without.append(data)
        
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


def arch2data(matrix, op_list):
    oper2feature = {'skip_connect': [1, 0, 0, 0], 'nor_conv_1x1': [0, 1, 0, 0], 'nor_conv_3x3': [0, 0, 1, 0], 
                        'avg_pool_3x3': [0, 0, 0, 1], 'none': [0, 0, 0, 0]}
    in_degrees = np.sum(matrix, axis=0)
    out_degrees = np.sum(matrix, axis=1)
    degrees = in_degrees + out_degrees
    if degrees[0] == 0:
        x0 = np.array([0, 0, 0, 0])
    else:
        x0 = (np.array(oper2feature[op_list[0]]) + np.array(oper2feature[op_list[1]]) + np.array(oper2feature[op_list[3]])) / degrees[0]
    if degrees[1] == 0:
        x1 = np.array([0, 0, 0, 0])
    else:
        x1 = (np.array(oper2feature[op_list[0]]) + np.array(oper2feature[op_list[2]]) + np.array(oper2feature[op_list[4]])) / degrees[1]
    if degrees[2] == 0:
        x2 = np.array([0, 0, 0, 0])
    else:
        x2 = (np.array(oper2feature[op_list[1]]) + np.array(oper2feature[op_list[2]]) + np.array(oper2feature[op_list[5]])) / degrees[2]
    if degrees[3] == 0:
        x3 = np.array([0, 0, 0, 0])
    else:
        x3 = (np.array(oper2feature[op_list[3]]) + np.array(oper2feature[op_list[4]]) + np.array(oper2feature[op_list[5]])) / degrees[3]
    x = np.vstack([x0, x1, x2, x3])
    x = torch.tensor(x, dtype=torch.float)
    indices = np.where(matrix == 1)
    indices = np.array(indices)
    edge_index = torch.tensor(indices, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    return data




if __name__ == '__main__':
    # setting
    EPOCH = 100
    batch_size = 64
    tiers_num = 3
    threshold = 0.5
    dataset = 'cifar10_valid'

    # save_path = r'./pkl/201_sampleset_200_{}.pkl'.format(dataset)
    save_path = r'./pkl/201_sampleset_100.pkl'
    with open(save_path, 'rb') as file:
        sampleset = pickle.load(file)
    data_sampled = sampleset.get_data_sampled(dataset)
    dataset = SiameseNetworkDataset(data_sampled=data_sampled, tiers_num=tiers_num, augmentation=False)

    num_split = 0.8
    num_train = int(len(dataset) * num_split)
    train_dataset = dataset[:num_train]
    test_dataset = dataset[num_train:]

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    SNet = SiameseNetwork(input_channels=4, hidden_channels=64, output_channels=2).to(device)
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
    # save_path = r'./predictor/Queries Number/cifar10_{}_201_SiameseNetwork_{}.pt'.format(90, suffix_num)
    save_path = r'./predictor/Queries Number/cifar10_{}_201_SiameseNetwork.pt'.format(100)
    torch.save(SNet.state_dict(), save_path)
    