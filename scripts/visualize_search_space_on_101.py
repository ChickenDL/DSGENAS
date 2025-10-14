'''
Descripttion: 
version: 1.0
Author: ZXL
Date: 2024-03-09 20:55:01
LastEditors: ZXL
LastEditTime: 2025-10-14 17:25:44
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
from get_data_from_101 import NASBench101, padding_zero_in_matrix
from scheme.nodeinfo_transmit import transmits
from torch_geometric.data import Data
from TSNetwork import TSNetwork
from TNetwork_on_101 import TNetwork
from SiameseNetwork_on_101 import SiameseNetwork


ModelSpec = _model_spec.ModelSpec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_path = os.path.dirname(__file__)
def arch2data(matrix, op_list):
    oper2feature = {'input': [1, 0, 0, 0, 0, 0], 'conv1x1-bn-relu': [0, 1, 0, 0, 0, 0], 'conv3x3-bn-relu': [0, 0, 1, 0, 0, 0], 
                        'maxpool3x3': [0, 0, 0, 1, 0, 0], 'null': [0, 0, 0, 0, 1, 0],'output':[0, 0, 0, 0, 0, 1]}
    x = [oper2feature[oper] for oper in op_list]
    x = torch.tensor(x, dtype=torch.float)
    indices = np.where(matrix == 1)
    indices = np.array(indices)
    edge_index = torch.tensor(indices, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    
    return data

def extract_encode(NASBench=None, encode_type='One-hot vector'):
    arch_list = []
    arch101_path = os.path.join(current_path+'/pkl', 'arch_on_101_by_{}.pkl'.format(encode_type))
    if not os.path.exists(arch101_path):
        nasbench101 = NASBench101()
        if encode_type == 'One-hot vector':
            for index in range(423624):
                fixed_stat, _ = nasbench101.get_info_by_index(index)
                matrix, op_list = fixed_stat['module_adjacency'], fixed_stat['module_operations']
                model_spec = ModelSpec(matrix, op_list)
                one_hot_vector = nasbench101.get_encode_by_model_spec(model_spec)
                arch_list.append(one_hot_vector)
        
        elif encode_type == 'transmit':
            for index in range(423624):
                fixed_stat, _ = nasbench101.get_info_by_index(index)
                matrix, op_list = fixed_stat['module_adjacency'], fixed_stat['module_operations']
                matrix, op_list = padding_zero_in_matrix(matrix, op_list)
                transmit_matrix = transmits(matrix, op_list)
                transmit_vector = transmit_matrix.flatten()
                arch_list.append(transmit_vector)
        elif encode_type == 'TNetwork-F':
            gcn = TNetwork(6, 64, 5).to(device)
            save_path = r'./predictor/PT/TNetwork-F_2024-04-23_19-26-35.pt'
            gcn.load_state_dict(torch.load(save_path))
            gcn.eval()
            for index in range(423624):
                fixed_stat, _ = nasbench101.get_info_by_index(index)
                matrix, op_list = fixed_stat['module_adjacency'], fixed_stat['module_operations']
                matrix, op_list = padding_zero_in_matrix(matrix, op_list)
                data = arch2data(matrix, op_list)
                data.to(device)
                f, _ = gcn(data.x, data.edge_index, data.batch)
                vector = f.view(-1).tolist()
                arch_list.append(vector)
                print(index)

        with open(arch101_path, 'wb') as file:
            pickle.dump(arch_list, file)

    return arch_list


def extract_acc(NASBench=None):
    acc_list = []
    acc101_path = os.path.join(current_path+'/pkl', 'acc_on_101.pkl')
    if not os.path.exists(acc101_path):
        nasbench101 = NASBench101()
        for index in range(423624):
            _ , computed_stat = nasbench101.get_info_by_index(index)
            mean_acc_list = []
            for i in range(3):
                mean_acc_list.append(computed_stat[108][i]['final_validation_accuracy'])
            mean_acc = np.mean(mean_acc_list)
            acc_list.append(mean_acc)
        with open(acc101_path, 'wb') as file:
            pickle.dump(acc_list, file)

    with open(acc101_path, 'rb') as file:
        acc_list = pickle.load(file)
    
    return acc_list


def partition_tier(acc_list, tier_num):
    sorted_acc_list = sorted(acc_list)
    cutpoint = int(423624 / 5)
    quintile = []
    tier_list = []
    for i in range(tier_num - 1):
        quintile.append(sorted_acc_list[cutpoint * (i + 1)])
    for acc in acc_list:
        if acc < quintile[0]:
            tier_list.append(0)
        elif acc >= quintile[0] and acc < quintile[1]:
            tier_list.append(1)
        elif acc >= quintile[1] and acc < quintile[2]:
            tier_list.append(2)
        elif acc >= quintile[2] and acc < quintile[3]:
            tier_list.append(3)
        else:
            tier_list.append(4)

    return tier_list


def visualize_by_tSNE(arch_list, acc_list, encode_type):
    arch_array = np.array(arch_list)
    acc_array = np.array(acc_list)
    random_state = 12 # seed
    tsne = TSNE(random_state=random_state, n_components=2, perplexity=20, init='pca')
    save_path = os.path.join(current_path+'/pkl', '{}_on_101_by_TSNE.pkl'.format(encode_type))
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
    cbar.set_ticks([1]+list(range(50000, 423624, 50000))+[423624])
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
    plt.savefig(current_path + '/figures/PDF-version/NAS-Bench-101 Search Space by {}.pdf'.format(encode_type))
    plt.savefig(current_path + '/figures/PNG-version/NAS-Bench-101 Search Space by {}.png'.format(encode_type))
    plt.show()


def visualize_Tier_by_tSNE(arch_list, acc_list, tier_list, encode_type):
    arch_array = np.array(arch_list)
    acc_array = np.array(acc_list)
    random_state = 12 # seed
    tsne = TSNE(random_state=random_state, n_components=2, perplexity=20, init='pca')
    save_path = os.path.join(current_path+'/pkl', '{}_on_101_by_TSNE.pkl'.format(encode_type))
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
    colors = ['black', '#008000', '#069AF3', 'orange', 'red']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=[0, 1, 2, 3, 4, 5], ncolors=cmap.N)
    metric = [colors[tier] for tier in tier_list]
    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.scatter(x, y, edgecolors='none', c=tier_list, s=20, alpha=0.5, cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_ticks([1, 2, 3, 4, 5])
    cbar.set_ticklabels(['tier1', 'tier2', 'tier3', 'tier4', 'tier5'])
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
    plt.savefig(current_path + '/figures/PDF-version/NAS-Bench-101 Search Space by Tier and TNetwork.pdf')
    plt.savefig(current_path + '/figures/PNG-version/NAS-Bench-101 Search Space by Tier and TNetwork.png')
    plt.show()


def visualize_offsprings_by_tSNE(arch_list, acc_list, encode_type):
    arch_array = np.array(arch_list)
    acc_array = np.array(acc_list)
    random_state = 12 # seed
    tsne = TSNE(random_state=random_state, n_components=2, perplexity=20, init='pca')
    save_path = os.path.join(current_path+'/pkl', '{}_on_101_by_TSNE.pkl'.format(encode_type))
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
    cbar.set_ticks([1]+list(range(50000, 423624, 50000))+[423624])
    cbar.set_label('Groud-Truth Ranking', size=20)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(20)
    cbar.ax.invert_yaxis()

    index_optimal = np.argmax(acc_array)
    x_optimal = X_embedded[index_optimal, 0]
    y_optimal = X_embedded[index_optimal, 1]
    im = ax.scatter(x_optimal, y_optimal, edgecolors='white', s=240, linewidth=0.4, color='red', alpha=1, marker='*', label='Optimal')
    ax.legend(loc='lower right', prop={'size': 20}, markerscale=1.2, handletextpad=-0.2)
    plt.savefig(current_path + '/figures/PDF-version/NAS-Bench-101 Search Space Offsprings by {}.pdf'.format(encode_type))
    plt.savefig(current_path + '/figures/PNG-version/NAS-Bench-101 Search Space Offsprings by {}.png'.format(encode_type))
    plt.show()




if __name__ == '__main__':
    type = 'TNetwork-F'
    arch_list = extract_encode(None, type)
    acc_list = extract_acc(None)
    # visualize_by_tSNE(arch_list, acc_list, type)
    # tier_list = partition_tier(acc_list, 5)
    nasbench101 = NASBench101()
    gcn = TNetwork(6, 64, 5).to(device)
    save_path = r'./predictor/PT/TNetwork-F_2024-04-23_19-26-35.pt'
    gcn.load_state_dict(torch.load(save_path))
    gcn.eval()
    # visualize_offsprings_by_tSNE(arch_list, acc_list, type)
    # tier_list = []
    # for index in range(423624):
    #     fixed_stat, _ = nasbench101.get_info_by_index(index)
    #     matrix, op_list = fixed_stat['module_adjacency'], fixed_stat['module_operations']
    #     matrix, op_list = padding_zero_in_matrix(matrix, op_list)
    #     data = arch2data(matrix, op_list)
    #     data.to(device)
    #     _, t = gcn(data.x, data.edge_index, data.batch)
    #     tier_list.append(t.cpu())
    #     print(index)
    visualize_by_tSNE(arch_list, acc_list, type)

    # tier_list_path = os.path.join(current_path+'/pkl', 'tier_list.pkl')
    # with open(tier_list_path, 'rb') as file:
    #     tier_list = pickle.load(file)
    # t_list = []
    # for tier in tier_list:
    #     _, t = torch.max(tier, 1)
    #     t_list.append(t)
    # visualize_Tier_by_tSNE(arch_list, acc_list, t_list, type)
    # save_path = r'./pkl/arch_on_101_hightier.pkl'
    # hightier_index = []
    # for index in range(423624):
    #     if tier_list[index] == 4:
    #         hightier_index.append(index)
    # with open(save_path, 'wb') as file:
    #     pickle.dump(hightier_index, file)
    