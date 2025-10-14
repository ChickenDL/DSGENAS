'''
Descripttion: 
version: 1.0
Author: ZXL
Date: 2023-11-21 14:55:44
LastEditors: ZXL
LastEditTime: 2025-10-14 17:22:56
'''
import numpy as np
import time
import datetime

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1_201 = 'nor_conv_1x1'
CONV3X3_201 = 'nor_conv_3x3'
MAXPOOL3X3 = 'maxpool3x3'
AVGPOOL3X3 = 'avg_pool_3x3'
NULL = 'null'


def get_newseed():
    return int(time.time())


def seed_log(seed):
    save_path = r'./output/seed_log/seed_{}.txt'.format(seed)
    with open(save_path, 'w') as file:
        file.write(str(seed))


def get_current_time():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return current_time

# matrix to upper triangular list
# in this process, the final upper triangular matrix does not contain diagonal elements of the original matrix
# therefore, there is no need to use delete_margin function in the subsequent encoding process
def matrix2utl(matrix):
    utl = []
    assert len(matrix) == len(matrix[0])
    matrix_len = len(matrix)
    for i in range(matrix_len - 1):
        utl += list(matrix[i][i + 1:])
    # flatten to 1-d
    utl = np.reshape(utl, -1)
    return utl


# upper triangular list to matrix
def utl2matrix(utl, matrix_len=7):
    matrix = np.zeros((matrix_len, matrix_len), dtype='int8')
    start_index = 0
    for i in range(matrix_len - 1):
        cur_len = matrix_len - i - 1
        matrix[i][i + 1:] = utl[start_index:start_index + cur_len]
        start_index += cur_len

    return matrix


def find_all_simple_paths(adj_matrix, start, end):
    def dfs(current, path):
        # 如果当前顶点已经在路径中，说明形成了环，返回
        if current in path:
            return
        # 将当前顶点加入路径
        path.append(current)
        # 如果当前顶点是目标顶点，记录路径
        if current == end:
            all_paths.append(list(path))
        else:
            # 遍历邻接矩阵的当前顶点的所有邻居
            for neighbor in range(len(adj_matrix)):
                if adj_matrix[current][neighbor] != 0:  # 有边相连
                    dfs(neighbor, path)
        # 回溯，移除当前顶点
        path.pop()

    all_paths = []
    dfs(start, [])
    return all_paths


def build_adj_matrix_from_paths(paths, num_nodes):
    # 初始化一个全零的邻接矩阵
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes), dtype='int8')

    # 遍历所有路径
    for path in paths:
        # 遍历路径中的每一对相邻节点
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            # 在邻接矩阵中设置对应位置为1
            adj_matrix[u][v] = 1

    return adj_matrix


# transform operation list to one-hot list
# input: op_list:str type
# output: one_hot_op_list:one_hot
def operations2onehot(op_list):
    dict_oper2one_hot = {NULL: [0, 0, 0, 1], CONV1X1: [0, 0, 1, 0], CONV1X1_201: [0, 0, 1, 0], CONV3X3: [0, 1, 0, 0], CONV3X3_201: [0, 1, 0, 0], MAXPOOL3X3: [1, 0, 0, 0], AVGPOOL3X3: [1, 0, 0, 0]}
    module_one_hot = np.array([dict_oper2one_hot[x] for x in op_list])
    # use [1: -1] to remove 'input' and 'output'
    module_one_hot = np.reshape(module_one_hot, (-1))
    module_one_hot = module_one_hot.tolist()
    return module_one_hot


# delete the first column and the last row in the matrix (because they are zeros)
# input: adjacent matrix (7*7)
# output: adjacent matrix (6*6)
def delete_margin(matrix):
    return matrix[:-1, 1:]


def population_log(gen_no, pops):
    save_path = r'./output/pops_log/gen_{}.txt'.format(gen_no)
    with open(save_path, 'w') as myfile:
        myfile.write(str(pops))
        myfile.write("\n")


def write_best_individual(gen_no, pops):
    arg_index = pops.get_sorted_index_order_by_acc()
    best_individual = pops.get_individual(arg_index[0])
    save_path = r'./output/pops_log/best_acc.txt'
    with open(save_path, 'a') as myfile:
        myfile.write('gen_no: {}'.format(gen_no) + '\n')
        myfile.write(str(best_individual))
        myfile.write('\n')


def sampleset_log(sampleset):
    current_time = get_current_time()
    save_path = r'./output/sample_log/sampleset_{}.txt'.format(current_time)
    with open(save_path, 'w') as myfile:
        myfile.write(str(sampleset))
        myfile.write("\n")


def GP_log(gen_no, query_pops, left_offspring):
    save_path = r'./output/pops_log/GP_{}.txt'.format(gen_no)
    with open(save_path, 'w') as myfile:
        myfile.write('query_pops\n')
        myfile.write(str(query_pops))
        myfile.write("\n")
        myfile.write('left_offspring\n')
        myfile.write(str(left_offspring))


def train_log(info, folder):
    save_path = r'./output/{}/train_log/epoch_loss_acc.txt'.format(folder)
    with open(save_path, 'a') as myfile:
        myfile.write(info)
        myfile.write('\n')


def write_category_center(center):
    save_path = r'./output/TNet_log/category_center.txt'
    with open(save_path, 'w') as myfile:
        for i in range(center.size()[0]):
            myfile.write('tier: {}'.format(i) + '\n')
            myfile.write(str(center[i]))
            myfile.write('\n')