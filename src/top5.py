'''
Descripttion: 
version: 1.0
Author: ZXL
Date: 2024-04-27 20:39:36
LastEditors: ZXL
LastEditTime: 2025-10-14 17:21:01
'''
import logging
import argparse
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from nas_201_api import NASBench201API as API201
from get_data_from_201 import NASBench201
from individual import IndividualY
from data_augmentation_HAAP import create_new_metrics
from SiameseNetwork_on_101 import SiameseNetwork as SiameseNetwork1
from SiameseNetwork_on_201 import SiameseNetwork as SiameseNetwork2
from nasbench.lib import model_spec as _model_spec
from get_data_from_101 import NASBench101, padding_zero_in_matrix

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NULL = 'null'
# nasbench = api.NASBench(NASBENCH_TFRECORD)
ModelSpec = _model_spec.ModelSpec
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def arch2data(matrix, op_list):
#     model_spec = ModelSpec(matrix, op_list)
#     pruned_matrix, pruned_op_list = model_spec.matrix, model_spec.ops
#     padding_matrix, padding_op_list = padding_zero_in_matrix(pruned_matrix, pruned_op_list)
#     oper2feature = {'input': [1, 0, 0, 0, 0, 0], 'conv1x1-bn-relu': [0, 1, 0, 0, 0, 0], 'conv3x3-bn-relu': [0, 0, 1, 0, 0, 0], 
#                         'maxpool3x3': [0, 0, 0, 1, 0, 0], 'null': [0, 0, 0, 0, 1, 0],'output':[0, 0, 0, 0, 0, 1]}
#     x = [oper2feature[oper] for oper in padding_op_list]
#     x = torch.tensor(x, dtype=torch.float)
#     indices = np.where(padding_matrix == 1)
#     indices = np.array(indices)
#     edge_index = torch.tensor(indices, dtype=torch.long)
#     data = Data(x=x, edge_index=edge_index)
    
#     return data

# x = 1
# print(x.shape)
# index1 = 200000
# index2 = 22572
# nasbench101 = NASBench101()
# fixed_stat1, _ = nasbench101.get_info_by_index(index1)
# matrix1 = fixed_stat1['module_adjacency']
# op_list1 = fixed_stat1['module_operations']
# print(nasbench101.get_info_by_index(index1))
# fixed_stat2, _ = nasbench101.get_info_by_index(index2)
# matrix2 = fixed_stat2['module_adjacency']
# op_list2 = fixed_stat2['module_operations']
# print(nasbench101.get_info_by_index(index2))
# data1 = arch2data(matrix1, op_list1)
# data2 = arch2data(matrix2, op_list2)
# SNet = SiameseNetwork(6, 64, 2)
# save_path = r'./predictor/PT/SiameseNetwork_2024-05-13_21-15-43.pt'
# SNet.load_state_dict(torch.load(save_path))
# SNet.eval()
# output = SNet(data1, data2)
# print(output)


# SNet = SiameseNetwork1(6, 64, 2).to(device)
# save_path = r'./predictor/PT/SiameseNetwork_2024-05-13_21-15-43.pt'
# SNet.load_state_dict(torch.load(save_path))
# SNet.eval()
# save_path = r'./pkl/arch_on_101_hightier.pkl'
# with open(save_path, 'rb') as file:
#     hightier_index = pickle.load(file)
# nasbench101 = NASBench101()
# anchor_index = 214722
# fixed_stat, _ = nasbench101.get_info_by_index(anchor_index)
# matrix = fixed_stat['module_adjacency']
# op_list = fixed_stat['module_operations']
# anchor = arch2data(matrix, op_list)
# anchor.to(device)
# flag = 0
# print(len(hightier_index))
# for index in hightier_index:
#     fixed_stat, _ = nasbench101.get_info_by_index(index)
#     matrix = fixed_stat['module_adjacency']
#     op_list = fixed_stat['module_operations']
#     data = arch2data(matrix, op_list)
#     data.to(device)
#     output = SNet(data, anchor)
#     _, predicted = torch.max(output, dim=1)
#     if predicted == 1:
#         flag += 1
# print(flag)

def arch_str2op_list(arch_str):
    op_list = []
    arch_str_list = API201.str2lists(arch_str)
    op_list.append(arch_str_list[0][0][0])
    op_list.append(arch_str_list[1][0][0])
    op_list.append(arch_str_list[1][1][0])
    op_list.append(arch_str_list[2][0][0])
    op_list.append(arch_str_list[2][1][0])
    op_list.append(arch_str_list[2][2][0])
    return op_list


def modify_matrix(op_str):
    matrix = np.array([[0, 1, 1, 1], 
                       [0, 0, 1, 1],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0]], dtype='int8')
    for i in range(6):
        if op_str[i] == 'none':
            if i == 0:
                matrix[0][1] = 0
            elif i == 1:
                matrix[0][2] = 0
            elif i == 2:
                matrix[1][2] = 0
            elif i == 3:
                matrix[0][3] = 0
            elif i == 4:
                matrix[1][3] = 0
            else:
                matrix[2][3] = 0

    return matrix


def arch2data(arch_str):
    op_list = arch_str2op_list(arch_str)
    matrix = modify_matrix(op_list)
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

dataset = 'ImageNet'

SNet = SiameseNetwork2(4, 64, 2).to(device)
save_path = r'./predictor/Queries Number/{}_200_201_SiameseNetwork.pt'.format(dataset)
SNet.load_state_dict(torch.load(save_path))
SNet.eval()
save_path = r'./pkl/ImageNet_arch_on_201_hightier.pkl'
with open(save_path, 'rb') as file:
    hightier_index = pickle.load(file)
nasbench201 = NASBench201()
anchor_index = 3888
info = nasbench201.get_info_by_index(anchor_index)
arch_str = info['arch_str']
anchor = arch2data(arch_str)
anchor.to(device)
flag = 0
print(len(hightier_index))
for index in hightier_index:
    info = nasbench201.get_info_by_index(index)
    arch_str = info['arch_str']
    data = arch2data(arch_str)
    data.to(device)
    output = SNet(data, anchor)
    _, predicted = torch.max(output, dim=1)
    if predicted == 1:
        flag += 1
print(flag)