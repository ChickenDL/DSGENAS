'''
Descripttion: 
version: 1.0
Author: ZXL
Date: 2023-12-07 19:30:33
LastEditors: ZXL
LastEditTime: 2025-09-18 15:05:41
'''
import json
import base64
import numpy as np
from nasbench.lib import config
from nasbench.lib import model_metrics_pb2
from nasbench.lib import model_spec as _model_spec
from utils import matrix2utl, utl2matrix, operations2onehot
import tensorflow._api.v2.compat.v1 as tf
import os
import pickle
import copy

# 加载数据
current_path = os.path.dirname(__file__)
NASBENCH_TFRECORD = current_path + "/nasbench/nasbench_only108.tfrecord"
NASBENCH_MAX_LEN = 423624
INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NULL = 'null'
# nasbench = api.NASBench(NASBENCH_TFRECORD)
ModelSpec = _model_spec.ModelSpec


class OutOfDomainError(Exception):
    """Throw an exception"""


# class with 101 data and related methods
class NASBench101(object):
    def __init__(self, data_path=None):
        if data_path is None:
            current_path = os.path.dirname(__file__)
            data_path = current_path + "/nasbench/nasbench_only108.tfrecord"
        self.config = config.build_config()
        # stores the fixed statistics that are independent of evaluation
        # adjacency matrix, operations, and number of parameters).
        # hash --> metric name --> scalar
        self.fixed_statistics = {}

        # stores the statistics that are computed via training and evaluating the
        # model on CIFAR-10. Statistics are computed for multiple repeats of each
        # model at each max epoch length.
        # hash --> epochs --> repeat index --> metric name --> scalar
        self.computed_statistics = {}

        self.hash_list = {}

        computed_statisticst_path = os.path.join(current_path+'/pkl','computed_statistics.pkl')
        if not os.path.exists(computed_statisticst_path):
            fixed_statistics = {}
            computed_statistics = {}
            # traverl files to read data
            for serialized_row in tf.python_io.tf_record_iterator(data_path):
                # parse the data from the data file.
                module_hash, epochs, raw_adjacency, raw_operations, raw_metrics = (
                    json.loads(serialized_row.decode('utf-8')))
                
                dim = int(np.sqrt(len(raw_adjacency)))
                adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
                adjacency = np.reshape(adjacency, (dim, dim))
                operations = raw_operations.split(',')
                metrics = model_metrics_pb2.ModelMetrics.FromString(
                    base64.b64decode(raw_metrics))

                # the architecture was trained three times, with a total of 423264 * 3 cycles
                # and different data needs to be considered for the architecture three times
                # the first iteration saves the architecture in the dictionary, and subsequent data with the architecture will be written back
                if module_hash not in computed_statistics:
                    # first time seeing this module, initialize fixed statistics.
                    new_entry = {}
                    new_entry['module_adjacency'] = adjacency
                    new_entry['module_operations'] = operations
                    new_entry['trainable_parameters'] = metrics.trainable_parameters
                    fixed_statistics[module_hash] = new_entry
                    computed_statistics[module_hash] = {}
                
                if epochs not in computed_statistics[module_hash]:
                    computed_statistics[module_hash][epochs] = [] 
                # each data_point consists of the metrics recorded from a single
                # train-and-evaluation of a model at a specific epoch length.
                data_point = {}

                # note: metrics.evaluation_data[0] contains the computed metrics at the
                # start of training (step 0) but this is unused by this API.

                # evaluation statistics at the end of training
                final_evaluation = metrics.evaluation_data[2]
                data_point['final_training_time'] = final_evaluation.training_time
                data_point['final_train_accuracy'] = final_evaluation.train_accuracy
                data_point['final_validation_accuracy'] = final_evaluation.validation_accuracy
                data_point['final_test_accuracy'] = final_evaluation.test_accuracy

                computed_statistics[module_hash][epochs].append(data_point)

            # save the loaded data to the pkl folder to accelerate the second load
            save_path = os.path.join(current_path + '/pkl', 'fixed_statistics.pkl')
            with open(save_path, 'wb') as file:
                pickle.dump(fixed_statistics, file)
            save_path = os.path.join(current_path + '/pkl','computed_statistics.pkl')
            with open(save_path,'wb') as file:
                pickle.dump(computed_statistics, file)
            print('fixed_statistics and computed_statistics successfully loaded')
        
        # load existing data from the pkl folder
        fixed_statistics_path = os.path.join(current_path + '/pkl', 'fixed_statistics.pkl')
        with open(fixed_statistics_path, 'rb') as file:
            self.fixed_statistics = pickle.load(file)
        computed_statisticst_path = os.path.join(current_path + '/pkl','computed_statistics.pkl')
        with open(computed_statisticst_path, 'rb') as file:
            self.computed_statistics = pickle.load(file)
        
        hash_list_path = os.path.join(current_path + '/pkl','hash_list.pkl')
        with open(hash_list_path, 'rb') as file:
            self.hash_list = pickle.load(file)

    def get_info_by_model_spec(self, model_spec):
        # self._check_spec(model_spec)
        model_hash = self._hash_spec(model_spec)
        return self.get_info_by_hash(model_hash)


    def get_info_by_hash(self, model_hash):
        fixed_stat = copy.deepcopy(self.fixed_statistics[model_hash])
        computed_stat = copy.deepcopy(self.computed_statistics[model_hash])
        return fixed_stat, computed_stat


    def get_info_by_index(self, index):
        return self.get_info_by_hash(self.hash_list[index])


    def get_encode_by_model_spec(self, model_spec):
        fixed_stat, _ = self.get_info_by_model_spec(model_spec)
        matrix = fixed_stat['module_adjacency']
        # exclude first and last nodes
        op_list = fixed_stat['module_operations']
        # determine if it is a 7x7 matrix, if not, pad it
        if len(op_list) != 7:
             matrix, op_list = padding_zero_in_matrix(matrix, op_list)
        utl = matrix2utl(matrix)
        onehot = operations2onehot(op_list[1:-1]) # operation2onehot function requires operations not include 'input' and 'output'
        encode = np.concatenate((utl, onehot))
        return encode


    def get_model_spec_by_encode(self, encode):
        utl = encode[:21]
        onehot = encode[21:]
        matrix = utl2matrix(utl)
        ops = []
        ops.append(INPUT)
        for i in range(len(onehot)):
            if onehot[i] == 1:
                if (i + 1) % 4 == 0:
                    ops.append(NULL)
                elif (i + 1) % 4 == 3:
                    ops.append(CONV1X1)
                elif (i + 1) % 4 == 2:
                    ops.append(CONV3X3)
                else:
                    ops.append(MAXPOOL3X3)
        ops.append(OUTPUT)

        # determine if it has been padded, if it has been padded, restore it
        if NULL in ops:
            matrix, ops = remove_zero_in_matrix(matrix, ops)
        # ValueError may be thrown here
        # Please note that, the model_spec.matrix is not the same with the matrix,
        # because the model_spec.matrix has pruned the matrix.
        model_spec = ModelSpec(matrix, ops)
        return model_spec


    def get_index_by_model_spec(self, model_spec):
        model_hash = self._hash_spec(model_spec)
        for index in range(NASBENCH_MAX_LEN):
            if self.hash_list[index] == model_hash:
                break
        return index


    def _check_spec(self, model_spec):
        if not model_spec.valid_spec:
            raise OutOfDomainError('invalid spec, provided gragh is diconnected.')

        num_vertices = len(model_spec.ops)
        num_edges = np.sum(model_spec.matrix)

        if num_vertices > self.config['module_vertices']:
            raise OutOfDomainError('too many vertics, got %d (max vertices = %d)'
                                    % (num_vertices, config['module_vertices']))
        if num_edges > self.config['max_edges']:
            raise OutOfDomainError('too many edges, got %d (max edges = %d)' 
                                   % (num_edges, self.config['max_edges']))

        if model_spec.ops[0] != 'input':
            raise OutOfDomainError('first operation should be \'input\'')
        if model_spec.ops[-1] != 'output':
            raise OutOfDomainError('last operation should be \'output\'')
        for op in model_spec.ops[1:-1]:
            if op not in self.config['available_ops']:
                raise OutOfDomainError('unsupported op %s (available ops = %s)'
                            % (op, self.config['available_ops']))
    
    
    def _hash_spec(self, model_spec):
        return model_spec.hash_spec(self.config['available_ops'])
    

# pad non 7x7 matrice
def padding_zero_in_matrix(module_adjacency, module_operations):
    len_operations = len(module_operations) # operations length
    for i in range(len_operations, 7):
        module_operations.insert(i-1, NULL)
    padding_operations = module_operations

    adjacent_matrix = module_adjacency
    padding_matrix = np.insert(adjacent_matrix, len_operations-1, np.zeros([7-len_operations, len_operations]), axis=0)
    padding_matrix = np.insert(padding_matrix, [len_operations-1], np.zeros([7, 7-len_operations]), axis=1)

    return padding_matrix, padding_operations
        

# prune padded matrice, restore to original size
def remove_zero_in_matrix(padding_matrix, padding_operations):
    module_operations = []
    for i in range(len(padding_operations)):
        if padding_operations[i] != NULL:
            module_operations.append(padding_operations[i])
        else:
            break
    module_operations.append(OUTPUT)

    module_matrix = np.delete(padding_matrix, slice(i, 6), axis=0)
    module_matrix = np.delete(module_matrix, slice(i, 6), axis=1)

    return module_matrix, module_operations


# def get_data_from_nasbenmark101(load_path):
#     # Stores the fixed statistics that are independent of evaluation
#     # adjacency matrix, operations, and number of parameters).
#     # hash --> metric name --> scalar
#     fixed_statistics = {}

#     # Stores the statistics that are computed via training and evaluating the
#     # model on CIFAR-10. Statistics are computed for multiple repeats of each
#     # model at each max epoch length.
#     # hash --> epochs --> repeat index --> metric name --> scalar
#     computed_statistics = {}

#     # traverl files to read data
#     for serialized_row in tf.python_io.tf_record_iterator(load_path):
#         # Parse the data from the data file.
#         module_hash, epochs, raw_adjacency, raw_operations, raw_metrics = (json.loads(serialized_row.decode('utf-8')))
        
#         dim = int(np.sqrt(len(raw_adjacency)))
#         adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
#         adjacency = np.reshape(adjacency, (dim, dim))
#         operations = raw_operations.split(',')
#         metrics = model_metrics_pb2.ModelMetrics.FromString(base64.b64decode(raw_metrics))

#         # The architecture was trained three times, with a total of 423264 * 3 cycles
#         # and different data needs to be considered for the architecture three times
#         # The first iteration saves the architecture in the dictionary, and subsequent data with the architecture will be written back
#         if module_hash not in computed_statistics:
#             # First time seeing this module, initialize fixed statistics.
#             new_entry = {}
#             new_entry['module_adjacency'] = adjacency
#             new_entry['module_operations'] = operations
#             new_entry['trainable_parameters'] = metrics.trainable_parameters
#             fixed_statistics[module_hash] = new_entry
#             computed_statistics[module_hash] = {}
        
#         if epochs not in computed_statistics[module_hash]:
#             computed_statistics[module_hash][epochs] = []   
#         # Each data_point consists of the metrics recorded from a single
#         # train-and-evaluation of a model at a specific epoch length.
#         data_point = {}

#         # Note: metrics.evaluation_data[0] contains the computed metrics at the
#         # start of training (step 0) but this is unused by this API.

#         # Evaluation statistics at the end of training
#         final_evaluation = metrics.evaluation_data[2]
#         data_point['final_training_time'] = final_evaluation.training_time
#         data_point['final_train_accuracy'] = final_evaluation.train_accuracy
#         data_point['final_validation_accuracy'] = final_evaluation.validation_accuracy
#         data_point['final_test_accuracy'] = final_evaluation.test_accuracy

#         computed_statistics[module_hash][epochs].append(data_point)
#     save_path = os.path.join(current_path + '/pkl', 'fixed_statistics.pkl')
#     with open(save_path, 'wb') as file:
#         pickle.dump(fixed_statistics, file)
#     save_path = os.path.join(current_path + '/pkl','computed_statistics.pkl')
#     with open(save_path,'wb') as file:
#         pickle.dump(computed_statistics, file)


# divide and conquer to find the top 1000 architectures
def calculate_mean_acc(x, index, type='test'): 
    final_accuracy_list = []
    for i in range(3):
        final_accuracy_list.append(x[index][108][i]['final_'+type+'_accuracy'])
    return np.mean(final_accuracy_list)


def partition(x, low, high, hash_list):
    pivot = calculate_mean_acc(x, hash_list[low])
    pivot_temp = x[hash_list[low]]
    while low < high:
        while low < high and calculate_mean_acc(x, hash_list[high]) >= pivot:
            high -= 1
        x[hash_list[low]] = x[hash_list[high]]
        while low < high and calculate_mean_acc(x, hash_list[low]) <= pivot:
            low += 1
        x[hash_list[high]] = x[hash_list[low]]
    x[hash_list[low]] = pivot_temp
    return low


def qsort(x, low, high, k, hash_list):
    if low < high:
        pivotloc = partition(x, low, high, hash_list)
        if high - pivotloc + 1 == k:
            return
        elif high - pivotloc + 1 < k:
            qsort(x, low, pivotloc-1, k-high+pivotloc-1, hash_list)
        else:
            qsort(x, pivotloc+1, high, k, hash_list)


def get_topk(topk, type='test'):
    computed_statisticst_path = os.path.join(current_path + '/pkl','computed_statistics.pkl')
    with open(computed_statisticst_path, 'rb') as file:
        computed_statisticst = pickle.load(file)
       
    # get index-model_hash list
    hash_list_path = os.path.join(current_path + '/pkl','hash_list.pkl')
    if not os.path.isfile(hash_list_path):
        hash_list = []
        for index, item in enumerate(computed_statisticst):
            hash_list.append(item)
        
        save_path = os.path.join(current_path + '/pkl','hash_list.pkl')
        with open(save_path,'wb') as file:
            pickle.dump(hash_list, file)
    
    with open(hash_list_path, 'rb') as file:
        hash_list = pickle.load(file)
    qsort(computed_statisticst, 0, NASBENCH_MAX_LEN-1, topk, hash_list)
    t = 0
    computed_statisticst_topk = {}
    while t < 1000:
        computed_statisticst_topk[hash_list[NASBENCH_MAX_LEN-1-t]] = computed_statisticst[hash_list[NASBENCH_MAX_LEN-1-t]]
        t += 1
    save_path = os.path.join(current_path + '/pkl','computed_statisticst_top{}.pkl'.format(topk))
    with open(save_path,'wb') as file:
        pickle.dump(computed_statisticst_topk, file)




# the following is the testing section
if __name__ == '__main__':
    # fixed_statistics_path = os.path.join(current_path + '/pkl', 'fixed_statistics.pkl')
    # with open(fixed_statistics_path, 'rb') as file:
    #     fixed_statistics = pickle.load(file)
    # computed_statisticst_path = os.path.join(current_path + '/pkl','computed_statistics.pkl')
    # with open(computed_statisticst_path, 'rb') as file:
    #     computed_statistics = pickle.load(file)
    # topk = 1000 # The first topk you need
    # computed_statisticst_top_path = os.path.join(current_path+'/pkl','computed_statisticst_top{}.pkl'.format(topk))
    # if not os.path.exists(computed_statisticst_top_path):
    #     get_topk(topk)
    #     print('computed_statisticst_top{} successfully loaded'.format(topk))
    nasbench101 = NASBench101()
    # print(nasbench101.get_info_by_index(22572))
    # fixed_stat, _ = nasbench101.get_info_by_index(23)
    # matrix = fixed_stat['module_adjacency']
    # print(matrix)
    # op_list = fixed_stat['module_operations']
    # print(op_list)
    # matrix, op_list = padding_zero_in_matrix(matrix, op_list)
    # print(matrix)
    # print(op_list)
    # utl = matrix2utl(matrix)
    # print(utl)
    # onehot = operations2onehot(op_list[1:-1])
    # print(onehot)
    # encode = np.concatenate((utl, onehot))
    # print(encode)
    # model_spec = nasbench101.get_model_spec_by_encode(encode)
    # print(nasbench101.get_info_by_model_spec(model_spec))y
    
    # hash_list_path = os.path.join(current_path + '/pkl','hash_list.pkl')
    # with open(hash_list_path, 'rb') as file:
    #     hash_list = pickle.load(file)
    # for i in range(NASBENCH_MAX_LEN):
    #     if len(nasbench101.fixed_statistics[hash_list[i]]['module_operations']) != 7:
    #         break
    # print(i)
    # find the best architeture
    # top5: 22572 181212 238047 110805 334673
    # max = 0
    # flag = 0
    # model_spec = ModelSpec(
    # # # Adjacency matrix of the module
    # matrix=[[0, 1, 1, 0, 0, 1, 1],
    #         [0, 0, 0, 0, 0, 1, 0],
    #         [0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 1, 0],
    #         [0, 0, 0, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 0, 0, 0]],
    # # Operations at the vertices of the module, matches order of matrix
    # ops=[INPUT, CONV1X1, CONV3X3, MAXPOOL3X3, CONV3X3, CONV3X3, OUTPUT])
    # print(nasbench101.get_info_by_model_spec(model_spec))
    # module_matrix = [[0, 1, 1, 1, 0],    # input layer
    #                  [0, 0, 0, 0, 1],    # 1x1 conv
    #                  [0, 0, 0, 0, 1],    # 3x3 conv
    #                  [0, 0, 0, 0, 0],    # 5x5 conv (replaced by two 3x3's)
    #                  [0, 0, 0, 0, 0]]    # output layer
    # module_operations = [INPUT, CONV1X1, CONV3X3, CONV3X3, OUTPUT]

    # padding_matrix, padding_operations = padding_zero_in_matrix(module_matrix, module_operations)
    # print(padding_matrix)
    # print(padding_operations)
    # module_matrix, module_operations = remove_zero_in_matrix(padding_matrix, padding_operations)
    # print(module_matrix)
    # print(module_operations)