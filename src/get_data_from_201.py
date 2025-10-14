'''
Descripttion: 
version: 1.0
Author: ZXL
Date: 2023-12-08 19:27:16
LastEditors: ZXL
LastEditTime: 2024-07-05 10:27:25
'''


import os
import pickle
import copy
import numpy as np
import collections
from nasbench.lib import model_spec as _model_spec
from nas_201_api import NASBench201API as API201
from utils import matrix2utl, utl2matrix, operations2onehot


ModelSpec = _model_spec.ModelSpec # Call the matrix-operation conversion function in the 101 API
# basic matrix for nas_bench 201
# Actually, this is a network structure with 6 edges in 201
# Then, convert it into an adjacency matrix corresponding to the 101 form
BASIC_MATRIX = [[0, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0]]

MAX_NUMBER = 15625

INPUT = 'input'
OUTPUT = 'output'
CONV1X1_201 = 'nor_conv_1x1'
CONV3X3_201 = 'nor_conv_3x3'
AVGPOOL3X3 = 'avg_pool_3x3'
NULL = 'null'

current_path = os.path.dirname(__file__)

# class with 201 data and related methods
class NASBench201(object):
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = current_path + '/nas_201_api/NAS-Bench-201-v1_1-096897.pth'

        self.arch_str2index = {}
        self.fixed_metrics = {}
        self.ordered_dict = collections.OrderedDict()
        
        index_list_path = os.path.join(current_path+'/pkl', 'index_list.pkl')
        tidy_file_path = os.path.join(current_path+'/pkl','tidy_file.pkl')
        if not os.path.exists(tidy_file_path):
            nasbench201 = API201(data_path)
            ordered_dict = collections.OrderedDict()
            arch_str2index = {}
            for index in range(len(nasbench201.evaluated_indexes)):
                info = nasbench201.query_meta_info_by_index(index, '200')
                arch_str = info.arch_str
                arch_str2index[arch_str] = index
                # cifar10, cifar100, ImageNet16-120
                cifar10_valid = info.get_metrics('cifar10-valid','x-valid')['accuracy']
                cifar10_test = info.get_metrics('cifar10','ori-test')['accuracy']
                cifar100_valid = info.get_metrics('cifar100','x-valid')['accuracy']
                cifar100_test = info.get_metrics('cifar100','x-test')['accuracy']
                ImageNet_valid = info.get_metrics('ImageNet16-120','x-valid')['accuracy']
                ImageNet_test = info.get_metrics('ImageNet16-120','x-test')['accuracy']
                index_info = {'arch_str': arch_str, 'cifar10_valid': cifar10_valid, 'cifar10_test': cifar10_test,
                            'cifar100_valid': cifar100_valid, 'cifar100_test': cifar100_test,
                            'ImageNet_valid': ImageNet_valid, 'ImageNet_test': ImageNet_test}
                ordered_dict[index] = index_info
            
            with open(index_list_path, 'wb') as file:
                pickle.dump(arch_str2index, file)
            with open(tidy_file_path, 'wb') as file:
                pickle.dump(ordered_dict, file)
        
        with open(index_list_path, 'rb') as file:
            self.arch_str2index = pickle.load(file)
        with open(tidy_file_path, 'rb') as file:
            self.ordered_dict = pickle.load(file)

        fixed_metrics_path = os.path.join(current_path + '/pkl', 'fixed_metrics.pkl')
        if not os.path.exists(fixed_metrics_path):
            fixed_metrics = {}
            for index in range(len(nasbench201.evaluated_indexes)):
                arch_str = self.ordered_dict[index]['arch_str']
                op_list = arch_str2op_list(arch_str)
                pruned_matrix, pruned_ops = delete_useless_node(op_list)
                if pruned_matrix is None:
                    index_info = {'module_adjacency': None, 'module_operations': None}
                    fixed_metrics[index] = index_info
                    continue
                if len(pruned_ops) != 8:
                    padding_matrix, padding_ops = padding_zeros(pruned_matrix, pruned_ops)
                    index_info = {'module_adjacency': padding_matrix, 'module_operations': padding_ops}
                else:
                    index_info = {'module_adjacency': pruned_matrix, 'module_operations': pruned_ops}
                fixed_metrics[index] = index_info
        
            with open(fixed_metrics_path, 'wb') as file:
                pickle.dump(fixed_metrics, file)

        with open(fixed_metrics_path, 'rb') as file:
            self.fixed_metrics = pickle.load(file)

    
    def get_info_by_index(self, index):
        info = self.ordered_dict[index]
        return info
    

    def get_info_by_arch_str(self, arch_str):
        info = self.get_info_by_index(self.arch_str2index[arch_str])
        return info
        

    def get_info_by_encode(self, encode):
        utl = encode[:28]
        onehot = encode[28:]
        matrix = utl2matrix(utl, 8)
        ops = []
        ops.append(INPUT)
        for i in range(len(onehot)):
            if onehot[i] == 1:
                if (i + 1) % 4 == 0:
                    ops.append(NULL)
                elif (i + 1) % 4 == 3:
                    ops.append(CONV1X1_201)
                elif (i + 1) % 4 == 2:
                    ops.append(CONV3X3_201)
                else:
                    ops.append(AVGPOOL3X3)
        ops.append(OUTPUT)
        # ValueError may be thrown here
        model_spec = ModelSpec(matrix, ops)
        padding_matrix, padding_operations = padding_zeros(model_spec.matrix, model_spec.ops)
        for index in range(MAX_NUMBER):
            if np.array_equal(padding_matrix, self.fixed_metrics[index]['module_adjacency']) and np.array_equal(padding_operations, self.fixed_metrics[index]['module_operations']):
                return self.get_info_by_index(index)
        
        return None

    def get_encode_by_arch_str(self, arch_str):
        index = self.arch_str2index[arch_str]
        padding_matrix = self.fixed_metrics[index]['module_adjacency']
        padding_ops = self.fixed_metrics[index]['module_operations']
        if padding_matrix is None:
            return None
        utl = matrix2utl(padding_matrix)
        onehot = operations2onehot(padding_ops[1:-1])
        encode = np.concatenate((utl, onehot))

        return encode
    
    
def delete_useless_node(ops):
    # delete the skip connections nodes and the none nodes
    # output the pruned metrics
    # start to change matrix
    matrix = copy.deepcopy(BASIC_MATRIX)
    for i, op in enumerate(ops, start=1):
        m = []
        n = []

        if op == 'skip_connect':
            for m_index in range(8):
                ele = matrix[m_index][i]
                if ele == 1:
                    # set element to 0
                    matrix[m_index][i] = 0
                    m.append(m_index)

            for n_index in range(8):
                ele = matrix[i][n_index]
                if ele == 1:
                    # set element to 0
                    matrix[i][n_index] = 0
                    n.append(n_index)

            for m_index in m:
                for n_index in n:
                    matrix[m_index][n_index] = 1

        elif op == 'none':
            for m_index in range(8):
                matrix[m_index][i] = 0
            for n_index in range(8):
                matrix[i][n_index] = 0

    ops_copy = copy.deepcopy(ops)
    ops_copy.insert(0, INPUT)
    ops_copy.append(OUTPUT)

    # start pruning
    model_spec = ModelSpec(matrix=matrix, ops=ops_copy)
    return model_spec.matrix, model_spec.ops


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


def padding_zeros(matrix, op_list):
    len_operations = len(op_list)
    if not len_operations == 8:
        for j in range(len_operations, 8):
            op_list.insert(j - 1, NULL)
        adjecent_matrix = copy.deepcopy(matrix)
        padding_matrix = np.insert(adjecent_matrix, len_operations - 1, np.zeros([8 - len_operations, len_operations]),
                                   axis=0)
        padding_matrix = np.insert(padding_matrix, [len_operations - 1], np.zeros([8, 8 - len_operations]), axis=1)

    return padding_matrix, op_list


def remove_zeros(padding_matrix, padding_operations):
    module_operations = []
    for i in range(len(padding_operations)):
        if padding_operations[i] != NULL:
            module_operations.append(padding_operations[i])
        else:
            break
    module_operations.append(OUTPUT)

    module_matrix = np.delete(padding_matrix, slice(i, 7), axis=0)
    module_matrix = np.delete(module_matrix, slice(i, 7), axis=1)

    print(module_matrix)
    return module_matrix, module_operations


def op_list2str(op_list):
    op_str = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(op_list[0], op_list[1], op_list[2], op_list[3], op_list[4],
                                                          op_list[5])
    return op_str

# def operation2integers(op_list):
#     dict_oper2int = {NULL: 0, CONV1X1: 1, CONV3X3: 2, AP3X3: 3}
#     module_integers = np.array([dict_oper2int[x] for x in op_list[1: -1]])
#     return module_integers




if __name__ == '__main__':
    nasbench201 = NASBench201()
    # info = nasbench201.get_info_by_index(6111)
    # print(info)
    # arch_str = info['arch_str']
    # print(arch_str)
    # encode = nasbench201.get_encode_by_arch_str(arch_str)
    # print(encode)
    # print(nasbench201.get_info_by_encode(encode))
    # arch_str = '|nor_conv_3x3~0|+|nor_conv_1x1~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_1x1~2|'
    # info = nasbench201.get_info_by_arch_str(arch_str)
    # print(info)
    # cifar10:         6111   91.61
    # cifar100:        9930   73.49
    # ImageNet:  10676  46.73

    # # current_path = os.path.dirname(__file__)
    # # data_path = current_path + '/nas_201_api/NAS-Bench-201-v1_1-096897.pth'
    # # api201 = API201(data_path)
    # op_list = arch_str2op_list(arch_str)
    # print(op_list)
    # pruned_matrix, pruned_op = delete_useless_node(op_list)
    # print(pruned_matrix, pruned_op)
    # padding_matrix, padding_op = padding_zeros(pruned_matrix, pruned_op)
    # print(padding_matrix, padding_op)
    # for index in range(MAX_NUMBER):
    #     info = nasbench201.get_info_by_index(index)
    #     arch_str = info['arch_str']
    #     print(arch_str)
    # op_integers = operation2integers(padding_op)
    # print(op_integers)
    # file_dict = torch.load(file_dict_path, map_location='cpu')
    # for index in range(len(nasbench201.evaluated_indexes())): 
    # file_dict['meta_archs'] -> meta_archs['index'] = arch
    # file_dict['arch2info'] -> arch2info['index']['hp'] = info
    # dataset = 'cifar10-valid' or 'cifar10' or 'cifar100' or 'ImageNet16-120'
    #  When dataset = cifar10-valid, you can use 'train', 'x-valid', 'ori-test'
    #       'train' : the metric on the training set.
    #       'x-valid' : the metric on the validation set.
    #       'ori-test' : the metric on the test set.
    #  When dataset = cifar10, you can use 'train', 'ori-test'.
    #       'train' : the metric on the training + validation set.
    #       'ori-test' : the metric on the test set.
    #  When dataset = cifar100 or ImageNet16-120, you can use 'train', 'ori-test', 'x-valid', 'x-test'
    #       'train' : the metric on the training set.
    #       'x-valid' : the metric on the validation set.
    #       'x-test' : the metric on the test set.
    #       'ori-test' : the metric on the validation + test set.
    # metric = info.get_compute_cost(dataset)
    # flop, param, latency = metric['flops'], metric['params'], metric['latency']
    # train_or_valid_or_test_info = info.get_metrics(dataset, 'train' or 'x-valid' or 'ori-test' or 'x-test')
    # train_or_valid_or_test_info['loss' or 'accuracy'][iepoch]
    # file_dict['evaluated_indexes'] -> evaluated_indexes\
    
