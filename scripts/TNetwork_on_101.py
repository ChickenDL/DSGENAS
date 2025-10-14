'''
Descripttion: 
version: 1.0
Author: ZXL
Date: 2024-03-11 16:45:25
LastEditors: ZXL
LastEditTime: 2025-10-14 17:13:12
'''
import random
import numpy as np
import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GraphConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from make_sample import SampleSetX
from nasbench.lib import model_spec as _model_spec
from data_augmentation_HAAP import create_new_metrics
from get_data_from_101 import padding_zero_in_matrix
from utils import train_log, get_current_time


ModelSpec = _model_spec.ModelSpec
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TNetwork(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(TNetwork, self).__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels*8)
        self.fc2 = nn.Linear(hidden_channels*8, hidden_channels*4)
        self.fc3 = nn.Linear(hidden_channels*4, hidden_channels*2)
        self.fc4 = nn.Linear(hidden_channels*2, hidden_channels)
        self.fc5 = nn.Linear(hidden_channels, output_channels)


    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        f = x

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc5(x)
        
        return f, x


class MarginContrastiveLoss(nn.Module):
    def __init__(self, margin=0.25):
        super(MarginContrastiveLoss, self).__init__()
        self.margin = margin
    
    
    def forward(self, embeddings, targets):
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        mask = targets.unsqueeze(1) == targets.unsqueeze(0)
        mask = mask.fill_diagonal_(False)
        positive_pairs = similarity_matrix[mask]
        negative_pairs = similarity_matrix[~mask]
        positive_loss = 1 - positive_pairs.mean()
        diff_matrix = torch.abs(targets.unsqueeze(1) - targets.unsqueeze(0))
        diff_matrix = diff_matrix[~mask]
        negative_loss = torch.clamp(self.margin * diff_matrix + negative_pairs, min=0.0).mean()
        loss_contrastive = positive_loss + negative_loss
        
        return loss_contrastive


class FusionLoss(nn.Module):
    def __init__(self, alpha=0.3):
        super(FusionLoss, self).__init__()
        self.alpha = alpha
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = MarginContrastiveLoss()


    def forward(self, embeddings, outputs, targets):
        loss_cls = self.cls_loss(outputs, targets)
        loss_reg = self.reg_loss(embeddings, targets)
        total_loss = loss_cls + self.alpha * loss_reg
        
        return total_loss


class AdaAlphaFusionLoss(nn.Module):
    def __init__(self):
        super(AdaAlphaFusionLoss, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 1)
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = MarginContrastiveLoss()

    
    def forward(self, embeddings, outputs, targets):
        loss_cls = self.cls_loss(outputs, targets)
        loss_reg = self.reg_loss(embeddings, targets)
        alpha = self.fc1(torch.cat((loss_cls.unsqueeze(0), loss_reg.unsqueeze(0))))
        alpha = F.relu(alpha)
        alpha = F.sigmoid(self.fc2(alpha))
        self.alpha = alpha

        total_loss = loss_cls + alpha * loss_reg

        return total_loss


class TNetworkDataset(Dataset):
    def __init__(self, data_sampled, tiers_num, augmentation=True):
        super(TNetworkDataset, self).__init__()
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
         # data augmentation
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
        # set label
        # Tier = []
        # tier_size = int(archsize_aug / self.tiers_num)
        sorted_data_sampled_aug = sorted(data_sampled_aug, key=lambda x:x['acc'])
        # for tier in range(self.tiers_num):
        #     start_index = tier * tier_size
        #     end_index = min((tier + 1) * tier_size, archsize_aug)
        #     Tier[start_index:end_index] = [tier] * (end_index - start_index)
    
        segment_size = archsize_aug // self.tiers_num
        remainder = archsize_aug % self.tiers_num
        segment_indices = [0] * archsize_aug
        current_segment = 1
        current_size = 0
        for index in range(archsize_aug):
            segment_indices[index] = current_segment
            current_size += 1
            if (current_segment <= remainder and current_size == segment_size + 1) or (current_segment > remainder and current_size == segment_size):
                current_segment += 1
                current_size = 0
        Tier = [x - 1 for x in segment_indices]
        # data_sampled -> torch_geometric.data, add node features
        
        data_aug = []
        oper2feature = {'input': [1, 0, 0, 0, 0, 0], 'conv1x1-bn-relu': [0, 1, 0, 0, 0, 0], 'conv3x3-bn-relu': [0, 0, 1, 0, 0, 0], 
                        'maxpool3x3': [0, 0, 0, 1, 0, 0], 'null': [0, 0, 0, 0, 1, 0], 'output':[0, 0, 0, 0, 0, 1]}
        for index in range(archsize_aug):
            sampled_arch = sorted_data_sampled_aug[index]
            arch = sampled_arch['arch']
            x = [oper2feature[oper] for oper in arch['op_list']]
            x = torch.tensor(x, dtype=torch.float)
            indices = np.where(arch['matrix'] == 1)
            indices = np.array(indices)
            edge_index = torch.tensor(indices, dtype=torch.long)
            y = Tier[index]
            data = Data(x=x, edge_index=edge_index, y=y)
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
        Tier = []
        tier_size = int(archsize / self.tiers_num)
        sorted_data_sampled = sorted(self.data_sampled, key=lambda x:x['acc'])
        for tier in range(self.tiers_num):
            start_index = tier * tier_size
            end_index = min((tier + 1) * tier_size, archsize)
            Tier[start_index:end_index] = [tier] * (end_index - start_index)
        
        data_without = []
        oper2feature = {'input': [1, 0, 0, 0, 0, 0], 'conv1x1-bn-relu': [0, 1, 0, 0, 0, 0], 'conv3x3-bn-relu': [0, 0, 1, 0, 0, 0], 
                        'maxpool3x3': [0, 0, 0, 1, 0, 0], 'null': [0, 0, 0, 0, 1, 0], 'output':[0, 0, 0, 0, 0, 1]}
        for index in range(archsize):
            sampled_arch = sorted_data_sampled[index]
            arch = sampled_arch['arch']
            x = [oper2feature[oper] for oper in arch['op_list']]
            indices = np.where(arch['matrix'] == 1)
            indices = np.array(indices)
            edge_index = torch.tensor(indices, dtype=torch.long)
            y = Tier[index]
            data = Data(x=x, edge_index=edge_index, y=y)
            data_without.append(data)
        return data_without
    

def train_TNetwork(model, EPOCH, train_loader, optimizer, criterion):
    model.train()
    for epoch in range(EPOCH):
        correct = 0
        running_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            embeddings, outputs = model(data.x, data.edge_index, data.batch)
            loss = criterion(embeddings, outputs, data.y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == data.y).sum().item()
        loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / len(train_loader.dataset)
        info = 'Epoch :  [{:03d}/{}]'.format(epoch+1, EPOCH) + '   |   loss :  {:08f}'.format(loss) + '   |   train_accuracy :  {:08f}%'.format(train_accuracy)
        train_log(info, 'TNet_log')
        print(info)

    return model


def train_TNetwork_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    correct = 0
    running_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        embeddings, outputs = model(data.x, data.edge_index, data.batch)
        loss = criterion(embeddings, outputs, data.y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == data.y).sum().item()
    loss = running_loss / len(train_loader)
    accuracy = 100 * correct / len(train_loader.dataset)

    return loss, accuracy


def test_TNetwork(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        for data in test_loader:
            data = data.to(device)
            _, outputs = model(data.x, data.edge_index, data.batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == data.y).sum().item()
    accuracy = 100 * correct / len(test_loader.dataset)
    
    return accuracy


def category_center(model, train_dataset, tiers_num):
    center = torch.zeros([tiers_num, 64]).to(device)
    num = torch.zeros(5).to(device)
    model.eval()
    with torch.no_grad():
        for data in train_dataset:
            data = data.to(device)
            embeddings, outputs = model(data.x, data.edge_index, data.batch)
            _, predicted = torch.max(outputs, 1)
            for index in range(len(predicted)):
                center[predicted[index]] += embeddings[index]
                num[predicted[index]] += 1

    avg_center = center / num.unsqueeze(1).expand(-1, 64)
    return avg_center





if __name__ == '__main__':
    # setting
    EPOCH = 300
    batch_size = 64
    tiers_num = 5

    save_path = r'./pkl/sampleset_100.pkl'
    with open(save_path, 'rb') as file:
        sampleset = pickle.load(file)
    data_sampled = sampleset.get_data_sampled()
    dataset = TNetworkDataset(data_sampled=data_sampled, tiers_num=tiers_num, augmentation=True)
    print(len(dataset))

    num_split = 0.8
    num_train = int(len(dataset) * num_split)
    train_dataset = dataset[:num_train]
    test_dataset = dataset[num_train:]

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
    # train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    TNet = TNetwork(dataset.num_node_features, hidden_channels=64, output_channels=dataset.num_classes).to(device)
    print(TNet)
    optimizer = optim.Adam(TNet.parameters())
    criterion = FusionLoss(0.6)
    # criterion = AdaAlphaFusionLoss().to(device)


    

    # train_TNetwork(TNet, EPOCH, train_loader, optimizer, criterion)
    for epoch in range(EPOCH):
        loss, train_accuracy = train_TNetwork_one_epoch(TNet, train_loader, optimizer, criterion)
        test_accuracy = test_TNetwork(TNet, test_loader)
        info1 = 'Epoch :  [{:03d}/{}]'.format(epoch+1, EPOCH) + '   |   loss :  {:08f}'.format(loss)
        info2 = '   |   train_accuracy :  {:08f}%'.format(train_accuracy) + '   |  test_accuracy :  {:08f}%'.format(test_accuracy)
        info = info1 + info2
        train_log(info, 'TNet_log')
        print(info)
    suffix_num = get_current_time()
    save_path = r'./predictor/Queries Number/{}_TNetwork-F_{}.pt'.format(100, suffix_num)
    torch.save(TNet.state_dict(), save_path)


