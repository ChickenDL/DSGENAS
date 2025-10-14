'''
Descripttion: 
version: 1.0
Author: ZXL
Date: 2024-03-09 20:55:01
LastEditors: ZXL
LastEditTime: 2025-10-14 17:28:05
'''
from __future__ import absolute_import
import os
import pickle
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from nasbench.lib import model_spec as _model_spec
from get_data_from_201 import NASBench201
from torch_geometric.data import Data
from TNetwork_on_201 import TNetwork
from nas_201_api import NASBench201API as API201

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_path = os.path.dirname(__file__)

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

def extract_encode(NASBench=None, encode_type='One-hot vector'):
    arch_list = []
    arch201_path = os.path.join(current_path+'/pkl', 'arch_on_201_by_{}.pkl'.format(encode_type))
    if not os.path.exists(arch201_path):
        nasbench201 = NASBench201()
        # encode_type == 'TNetwork-F'
        gcn = TNetwork(4, 64, 3).to(device)
        save_path = r'./predictor/PT/201_TNetwork-F_2024-05-15_14-44-31.pt'
        gcn.load_state_dict(torch.load(save_path))
        gcn.eval()
        for index in range(15625):
            info = nasbench201.get_info_by_index(index)
            arch_str = info['arch_str']
            data = arch2data(arch_str)
            data.to(device)
            f, _ = gcn(data.x, data.edge_index, data.batch)
            vector = f.view(-1).tolist()
            arch_list.append(vector)
            print(index)

        with open(arch201_path, 'wb') as file:
            pickle.dump(arch_list, file)

    with open(arch201_path, 'rb') as file:
        arch_list = pickle.load(file)

    return arch_list


def extract_acc(dataset='cifar10_valid'):
    acc_list = []
    acc201_path = os.path.join(current_path+'/pkl', '{}-acc_on_201.pkl'.format(dataset))
    if not os.path.exists(acc201_path):
        nasbench201 = NASBench201()
        for index in range(15625):
            info = nasbench201.get_info_by_index(index)
            acc = info[dataset]
            acc_list.append(acc)
        with open(acc201_path, 'wb') as file:
            pickle.dump(acc_list, file)

    with open(acc201_path, 'rb') as file:
        acc_list = pickle.load(file)
    
    return acc_list


def partition_tier(acc_list, tier_num):
    sorted_acc_list = sorted(acc_list)
    cutpoint = int(15625 / tier_num)
    quintile = []
    tier_list = []
    for i in range(tier_num - 1):
        quintile.append(sorted_acc_list[cutpoint * (i + 1)])
    for acc in acc_list:
        if acc < quintile[0]:
            tier_list.append(0)
        elif acc >= quintile[0] and acc < quintile[1]:
            tier_list.append(1)
        else:
            tier_list.append(2)

    return tier_list


def visualize_by_tSNE(arch_list, acc_list, encode_type):
    arch_array = np.array(arch_list)
    acc_array = np.array(acc_list)
    random_state = 12 # seed
    tsne = TSNE(random_state=random_state, n_components=2, perplexity=5, init='pca')
    save_path = os.path.join(current_path+'/pkl', '{}_on_201_by_TSNE.pkl'.format(encode_type))
    if not os.path.exists(save_path):
        X_embedded = tsne.fit_transform(arch_array) # dimensionality reduction
        with open(save_path, 'wb') as file:
            pickle.dump(X_embedded, file)
    with open(save_path, 'rb') as file:
        X_embedded = pickle.load(file)
    x = X_embedded[:,0]
    y = X_embedded[:,1]   
    RANK = (-np.array(acc_array)).argsort().argsort() + 1 
    metric = RANK # use ranking to represent colors
    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.scatter(x, y, edgecolors='none', c=metric, s=20, alpha=0.5, cmap='viridis_r')
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_ticks([1]+list(range(5000, 15625, 5000))+[15625])
    cbar.set_label('Groud-Truth Ranking', size=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(20)
    cbar.ax.invert_yaxis()

    index_optimal = np.argmax(acc_array)
    x_optimal = X_embedded[index_optimal, 0]
    y_optimal = X_embedded[index_optimal, 1]
    im = ax.scatter(x_optimal, y_optimal, edgecolors='white', s=240, linewidth=0.4, color='red', alpha=1, marker='*', label='Optimal')
    ax.legend(loc='lower right', prop={'size': 20}, markerscale=1.2, handletextpad=-0.2)
    plt.savefig(current_path + '/figures/PDF-version/NAS-Bench-201 Search Space by {}.pdf'.format(encode_type))
    plt.savefig(current_path + '/figures/PNG-version/NAS-Bench-201 Search Space by {}.png'.format(encode_type))
    plt.show()


def visualize_Tier_by_tSNE(arch_list, acc_list, tier_list, encode_type):
    arch_array = np.array(arch_list)
    acc_array = np.array(acc_list)
    random_state = 12 # seed
    tsne = TSNE(random_state=random_state, n_components=2, perplexity=20, init='pca')
    save_path = os.path.join(current_path+'/pkl', '{}_on_201_by_TSNE.pkl'.format(encode_type))
    if not os.path.exists(save_path):
        X_embedded = tsne.fit_transform(arch_array) # dimensionality reduction
        with open(save_path, 'wb') as file:
            pickle.dump(X_embedded, file)
    with open(save_path, 'rb') as file:
        X_embedded = pickle.load(file)
    x = X_embedded[:,0]
    y = X_embedded[:,1]
    # RANK = (-np.array(acc_array)).argsort().argsort() + 1 
    # metric = RANK # use ranking to represent colors
    metric = []
    colors = ['black', 'orange', 'red']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=[0, 1, 2, 3], ncolors=cmap.N)
    metric = [colors[tier] for tier in tier_list]
    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.scatter(x, y, edgecolors='none', c=tier_list, s=20, alpha=0.5, cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['tier1', 'tier2', 'tier3'])
    cbar.set_label('Predicted Tier', size=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(20)
    cbar.ax.invert_yaxis()

    index_optimal = np.argmax(acc_array)
    x_optimal = X_embedded[index_optimal, 0]
    y_optimal = X_embedded[index_optimal, 1]
    im = ax.scatter(x_optimal, y_optimal, edgecolors='white', s=240, linewidth=0.4, color='red', alpha=1, marker='*', label='Optimal')
    ax.legend(loc='lower right', prop={'size': 20}, markerscale=1.2, handletextpad=-0.2)
    plt.savefig(current_path + '/figures/PDF-version/NAS-Bench-201 Search Space by Tier and TNetwork.pdf')
    plt.savefig(current_path + '/figures/PNG-version/NAS-Bench-201 Search Space by Tier and TNetwork.png')
    plt.show()




if __name__ == '__main__':
    type = 'TNetwork-F'
    dataset = 'cifar10_valid'
    arch_list = extract_encode(None, type)
    acc_list = extract_acc(dataset)
    # visualize_by_tSNE(arch_list, acc_list, type)
    tier_list = partition_tier(acc_list, 3)
    nasbench201 = NASBench201()
    gcn = TNetwork(4, 64, 3).to(device)
    save_path = r'./predictor/PT/201_TNetwork-F_2024-05-15_14-44-31.pt'
    gcn.load_state_dict(torch.load(save_path))
    gcn.eval()
    tier_list = []
    # for index in range(15625):
    #     info = nasbench201.get_info_by_index(index)
    #     arch_str = info['arch_str']
    #     data = arch2data(arch_str)
    #     data.to(device)
    #     _, t = gcn(data.x, data.edge_index, data.batch)
    #     tier_list.append(t.cpu())
    #     print(index)
    tier_list_path = os.path.join(current_path+'/pkl', '201_tier_list.pkl')
    visualize_by_tSNE(arch_list, acc_list, type)

    # with open(tier_list_path, 'wb') as file:
    #     pickle.dump(tier_list, file)
    # with open(tier_list_path, 'rb') as file:
    #     tier_list = pickle.load(file)
    # t_list = []
    # for tier in tier_list:
    #     _, t = torch.max(tier, 1)
    #     t_list.append(t)
    # visualize_Tier_by_tSNE(arch_list, acc_list, t_list, type)
    # save_path = r'./pkl/ImageNet_arch_on_201_hightier{}.pkl'.format(dataset)
    # hightier_index = []
    # for index in range(15625):
    #     if t_list[index] == 2:
    #         hightier_index.append(index)
    # with open(save_path, 'wb') as file:
    #     pickle.dump(hightier_index, file)