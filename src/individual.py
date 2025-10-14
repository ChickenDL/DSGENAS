import numpy as np
import random
from nasbench.lib import model_spec as _model_spec
from utils import utl2matrix, matrix2utl
import copy
from get_data_from_101 import NASBench101, padding_zero_in_matrix


ModelSpec = _model_spec.ModelSpec
nasbench101 = NASBench101()

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
CONV1X1_201 = 'nor_conv_1x1'
CONV3X3_201 = 'nor_conv_3x3'
AVGPOOL3X3 = 'avg_pool_3x3'
NULL = 'null'

NONE = 'none'
SKIP_CONNECT = 'skip_connect'
NOR_CONV1X1 = 'nor_conv_1x1'
NOR_CONV3X3 = 'nor_conv_3x3'
AVGPOOL3X3 = 'avg_pool_3x3'

class IndividualX:
    def __init__(self, m_num_matrix=1, m_num_op_list=1):
        self.indi = {}
        self.m_num_matrix = m_num_matrix
        self.m_num_op_list = m_num_op_list
        self.mean_acc = 0
        self.random_acc = 0
        self.test_mean_acc = 0
    
    
    def clear_state_info(self):
        self.mean_acc = 0
        self.random_acc = 0
        self.test_mean_acc = 0


    def create_an_individual(self, matrix, op_list):
        self.indi['matrix'] = matrix
        self.indi['op_list'] = op_list

    
    def initialize(self):
        self.indi['matrix'], self.indi['op_list'] = self.init_one_individual()


    def init_one_individual(self):
        # initial op_list
        op_list = []
        op_list.append(INPUT)
        for _ in range(5):
            rand_int = random.randint(1, 3)
            if rand_int == 1:
                op_list.append(CONV1X1)
            elif rand_int == 2:
                op_list.append(CONV3X3)
            else:  # rand_int==3
                op_list.append(MAXPOOL3X3)
        op_list.append(OUTPUT)

        matrix = np.zeros(shape=(7, 7), dtype='int8')
        model_spec = ModelSpec(matrix=matrix, ops=op_list)

        # if the matrix contains more than nine edges or the matrix is invalid (the graph is disconnected),
        # then reinitialize
        # please note that, the model_spec.matrix is not the same with the matrix,
        # because the model_spec.matrix has pruned the matrix
        while (not model_spec.valid_spec) or (np.sum(model_spec.matrix) > 9):
            # print('Start to initial a individual')
            matrix = np.zeros(shape=(7, 7), dtype='int8')
            # initial matrix by row, and the matrix must be upper triangular
            # the first row must contains at least one 1, and the last row must be all zeros
            # the middle five rows and have a 1/4 probability to be all zeros, and otherwise, it must contain
            # at least one 1
            # row 0 to 5
            for i in range(0, 6):
                if random.random() < 0.75 or i == 0:
                    num_ones = random.randint(1, 6 - i)
                    one_index = random.sample(range(1 + i, 7), num_ones)
                    for j in one_index:
                        matrix[i][j] = 1
            model_spec = ModelSpec(matrix=matrix, ops=op_list)

        return matrix, op_list
    

    def init_one_individual_by_random_sample(self):
        index = random.randint(0, 423623)
        fixed_stat, _ = nasbench101.get_info_by_index(index)
        matrix, op_list = fixed_stat['module_adjacency'], fixed_stat['module_operations']
        padding_matrix, padding_op_list = padding_zero_in_matrix(matrix, op_list)

        return padding_matrix, padding_op_list


    def set_mean_acc(self, mean_acc):
        self.mean_acc = mean_acc


    def set_random_acc(self, random_acc):
        self.random_acc = random_acc


    def set_test_mean_acc(self, test_mean_acc):
        self.test_mean_acc = test_mean_acc


    def mutation(self):
        # self.matrix_mutation()
        self.op_list_mutation()


    def matrix_mutation(self):
        def point_flip(point):
            if point == 0:
                return 1
            else:  # point==1
                return 0

        # avoid produce invalid matrix
        while True:
            # sample the flip points from 21 positions
            flip_positions = random.sample(range(21), self.m_num_matrix)
            utl = matrix2utl(self.indi['matrix'])
            for index in flip_positions:
                utl[index] = point_flip(utl[index])
            matrix = utl2matrix(utl)
            model_spec = ModelSpec(matrix=matrix, ops=self.indi['op_list'])
            if model_spec.valid_spec and (np.sum(model_spec.matrix) <= 9):
                break

        self.indi['matrix'] = utl2matrix(utl)

    
    def op_list_mutation(self):
        mutation_positions = random.sample(range(1, 6), self.m_num_op_list)
        op_list = copy.deepcopy(self.indi['op_list'])
        for index in mutation_positions:
            cur_op = op_list[index]
            if cur_op == CONV1X1:
                new_op = random.choice([CONV3X3, MAXPOOL3X3]) # ensure that the mutation produces operations that are different from the original operation
            elif cur_op == CONV3X3:
                new_op = random.choice([CONV1X1, MAXPOOL3X3])
            elif cur_op == MAXPOOL3X3:
                new_op = random.choice([CONV1X1, CONV3X3])
            else: # cur_op == NULL
                new_op = random.choice([CONV1X1, CONV3X3, MAXPOOL3X3])
            # else:
            #     raise ValueError(
            #         'The op should be in [CONV1X1, CONV3X3, MAXPOOL3X3], but it is: {}'.format(cur_op))
            op_list[index] = new_op
        self.indi['op_list'] = op_list

    
    def __str__(self):
        str_ = []
        str_.append('Matrix:{}, Op_list:{}'.format(self.indi['matrix'], self.indi['op_list']))
        str_.append('Mean_ACC:{:.16f}'.format(self.mean_acc))
        str_.append('Random_ACC:{:.16f}'.format(self.random_acc))
        str_.append('Test_Mean_ACC:{:.16f}'.format(self.test_mean_acc))
        return ', '.join(str_)                                                    # 按要求打印indi


# class IndividualY:
#     def __init__(self, m_num_matrix=1, m_num_op_list=1):
#         self.indi = {}
#         self.m_num_matrix = m_num_matrix
#         self.m_num_op_list = m_num_op_list
#         self.mean_acc = {}
#         self.mean_acc['cifar10_valid'] = 0
#         self.mean_acc['cifar10_test'] = 0
#         self.mean_acc['cifar100_valid'] = 0
#         self.mean_acc['cifar100_test'] = 0
#         self.mean_acc['ImageNet_valid'] = 0
#         self.mean_acc['ImageNet_test'] = 0

    
#     def clear_state_info(self):
#         self.mean_acc['cifar10_valid'] = 0
#         self.mean_acc['cifar10_test'] = 0
#         self.mean_acc['cifar100_valid'] = 0
#         self.mean_acc['cifar100_test'] = 0
#         self.mean_acc['ImageNet_valid'] = 0
#         self.mean_acc['ImageNet_test'] = 0


#     def create_an_individual(self, matrix, op_list):
#         self.indi['matrix'] = matrix
#         self.indi['op_list'] = op_list

    
#     def initialize(self):
#         self.indi['matrix'], self.indi['op_list'] = self.init_one_individual()


#     def init_one_individual(self):
#         # initial op_list
#         op_list = []
#         op_list.append(INPUT)
#         for _ in range(6):
#             rand_int = random.randint(1, 3)
#             if rand_int == 1:
#                 op_list.append(CONV1X1_201)
#             elif rand_int == 2:
#                 op_list.append(CONV3X3_201)
#             else:  # rand_int==3
#                 op_list.append(AVGPOOL3X3)
#         op_list.append(OUTPUT)

#         matrix = np.zeros(shape=(8, 8), dtype='int8')
#         model_spec = ModelSpec(matrix=matrix, ops=op_list)

#         # if the matrix contains more than nine edges or the matrix is invalid (the graph is disconnected),
#         # then reinitialize
#         # please note that, the model_spec.matrix is not the same with the matrix,
#         # because the model_spec.matrix has pruned the matrix
#         while (not model_spec.valid_spec) or (np.sum(model_spec.matrix) > 9):
#             # print('Start to initial a individual')
#             matrix = np.zeros(shape=(8, 8), dtype='int8')
#             # initial matrix by row, and the matrix must be upper triangular
#             # the first row must contains at least one 1, and the last row must be all zeros
#             # the middle five rows and have a 1/4 probability to be all zeros, and otherwise, it must contain
#             # at least one 1
#             # row 0 to 6
#             for i in range(0, 7):
#                 if random.random() < 0.75 or i == 0:
#                     num_ones = random.randint(1, 7 - i)
#                     one_index = random.sample(range(1 + i, 8), num_ones)
#                     for j in one_index:
#                         matrix[i][j] = 1
#                 # else, this row are all zeros
#             model_spec = ModelSpec(matrix=matrix, ops=op_list)

#         return matrix, op_list
    

#     def set_mean_acc(self, mean_acc):
#         self.mean_acc = mean_acc


#     def mutation(self):
#         self.matrix_mutation()
#         self.op_list_mutation()


#     def matrix_mutation(self):
#         def point_flip(point):
#             if point == 0:
#                 return 1
#             else:  # point==1
#                 return 0

#         # avoid produce invalid matrix
#         while True:
#             # sample the flip points from 21 positions
#             flip_positions = random.sample(range(28), self.m_num_matrix)
#             utl = matrix2utl(self.indi['matrix'])
#             for index in flip_positions:
#                 utl[index] = point_flip(utl[index])
#             matrix = utl2matrix(utl, 8)
#             model_spec = ModelSpec(matrix=matrix, ops=self.indi['op_list'])
#             if model_spec.valid_spec and (np.sum(model_spec.matrix) <= 9):
#                 break

#         self.indi['matrix'] = utl2matrix(utl, 8)

    
#     def op_list_mutation(self):
#         mutation_positions = random.sample(range(1, 7), self.m_num_op_list)
#         op_list = copy.deepcopy(self.indi['op_list'])
#         for index in mutation_positions:
#             cur_op = op_list[index]
#             if cur_op == CONV1X1_201:
#                 new_op = random.choice([CONV3X3_201, AVGPOOL3X3]) # ensure that the mutation produces operations that are different from the original operation
#             elif cur_op == CONV3X3_201:
#                 new_op = random.choice([CONV1X1_201, AVGPOOL3X3])
#             elif cur_op == AVGPOOL3X3:
#                 new_op = random.choice([CONV1X1_201, CONV3X3_201])
#             # elif cur_op == NULL:
#             #     new_op = random.choice([CONV1X1, CONV3X3, MAXPOOL3X3])
#             else:
#                 raise ValueError(
#                     'The op should be in [CONV1X1, CONV3X3, MAXPOOL3X3], but it is: {}'.format(cur_op))
#             op_list[index] = new_op
#         self.indi['op_list'] = op_list

    
#     def __str__(self):
#         str_ = []
#         str_.append('Matrix:{}, Op_list:{}'.format(self.indi['matrix'], self.indi['op_list']))
#         str_.append('Mean_ACC in cifar10_valid:{:.16f}'.format(self.mean_acc['cifar10_valid']))
#         str_.append('Mean_ACC in cifar10_test:{:.16f}'.format(self.mean_acc['cifar10_test']))
#         str_.append('Mean_ACC in cifar100_valid:{:.16f}'.format(self.mean_acc['cifar100_valid']))
#         str_.append('Mean_ACC in cifar100_test:{:.16f}'.format(self.mean_acc['cifar100_test']))
#         str_.append('Mean_ACC in ImageNet_valid:{:.16f}'.format(self.mean_acc['ImageNet_valid']))
#         str_.append('Mean_ACC in ImageNet_test:{:.16f}'.format(self.mean_acc['ImageNet_test']))
#         return ', '.join(str_) 


class IndividualY:
    def __init__(self, m_num_matrix=1, m_num_op_list=1):
        self.indi = {}
        self.m_num_matrix = m_num_matrix
        self.m_num_op_list = m_num_op_list
        self.mean_acc = {}
        self.mean_acc['cifar10_valid'] = 0
        self.mean_acc['cifar10_test'] = 0
        self.mean_acc['cifar100_valid'] = 0
        self.mean_acc['cifar100_test'] = 0
        self.mean_acc['ImageNet_valid'] = 0
        self.mean_acc['ImageNet_test'] = 0

    
    def clear_state_info(self):
        self.mean_acc['cifar10_valid'] = 0
        self.mean_acc['cifar10_test'] = 0
        self.mean_acc['cifar100_valid'] = 0
        self.mean_acc['cifar100_test'] = 0
        self.mean_acc['ImageNet_valid'] = 0
        self.mean_acc['ImageNet_test'] = 0


    def create_an_individual(self, matrix, op_list):
        self.indi['matrix'] = matrix
        self.indi['op_list'] = op_list

    
    def initialize(self):
        self.indi['matrix'], self.indi['op_list'] = self.init_one_individual()


    def init_one_individual(self):
        matrix = np.array([[0, 1, 1, 1], 
                           [0, 0, 1, 1],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0]], dtype='int8')
        # initial op_list
        op_list = []
        for i in range(6):
            rand_int = random.randint(1, 5)
            if rand_int == 1:
                op_list.append(NONE)
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
            elif rand_int == 2:
                op_list.append(SKIP_CONNECT)
            elif rand_int == 3:
                op_list.append(NOR_CONV1X1)
            elif rand_int == 4:
                op_list.append(NOR_CONV3X3)
            else:  # rand_int==3
                op_list.append(AVGPOOL3X3)

        return matrix, op_list
    

    def set_mean_acc(self, mean_acc):
        self.mean_acc = mean_acc


    def mutation(self):
        self.op_list_mutation()

    
    def op_list_mutation(self):
        mutation_positions = random.sample(range(1, 6), self.m_num_op_list)
        op_list = copy.deepcopy(self.indi['op_list'])
        for index in mutation_positions:
            cur_op = op_list[index]
            if cur_op == NONE:
                new_op = random.choice([SKIP_CONNECT, NOR_CONV1X1, NOR_CONV3X3, AVGPOOL3X3]) # ensure that the mutation produces operations that are different from the original operation
            elif cur_op == SKIP_CONNECT:
                new_op = random.choice([NONE, NOR_CONV1X1, NOR_CONV3X3, AVGPOOL3X3])
            elif cur_op == NOR_CONV1X1:
                new_op = random.choice([NONE, SKIP_CONNECT, NOR_CONV3X3, AVGPOOL3X3])
            elif cur_op == NOR_CONV3X3:
                new_op = random.choice([NONE, SKIP_CONNECT, NOR_CONV1X1, AVGPOOL3X3])
            elif cur_op == AVGPOOL3X3:
                new_op = random.choice([NONE, SKIP_CONNECT, NOR_CONV1X1, NOR_CONV3X3])
            else:
                raise ValueError(
                    'The op should be in [CONV1X1, CONV3X3, MAXPOOL3X3], but it is: {}'.format(cur_op))
            op_list[index] = new_op
        matrix = np.array([[0, 1, 1, 1], 
                           [0, 0, 1, 1],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0]], dtype='int8')
        for i in range(6):
            if op_list[i] == NONE:
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
        self.indi['matrix'] = matrix
        self.indi['op_list'] = op_list

    
    def __str__(self):
        str_ = []
        str_.append('Matrix:{}, Op_list:{}'.format(self.indi['matrix'], self.indi['op_list']))
        str_.append('Mean_ACC in cifar10_valid:{:.16f}'.format(self.mean_acc['cifar10_valid']))
        str_.append('Mean_ACC in cifar10_test:{:.16f}'.format(self.mean_acc['cifar10_test']))
        str_.append('Mean_ACC in cifar100_valid:{:.16f}'.format(self.mean_acc['cifar100_valid']))
        str_.append('Mean_ACC in cifar100_test:{:.16f}'.format(self.mean_acc['cifar100_test']))
        str_.append('Mean_ACC in ImageNet_valid:{:.16f}'.format(self.mean_acc['ImageNet_valid']))
        str_.append('Mean_ACC in ImageNet_test:{:.16f}'.format(self.mean_acc['ImageNet_test']))
        return ', '.join(str_) 




if __name__ == '__main__':
    # indi = IndividualX()
    # indi.initialize()
    # print(indi)

    indi = IndividualY(3,3)
    indi.initialize()
    print(indi)
    indi.mutation()
    print(indi)
    indi.mutation()
    print(indi)
    # matrix = [[0, 1, 0, 0, 0, 1, 0], 
    #           [0, 0, 0, 0, 1, 1, 0], 
    #           [0, 0, 0, 1, 1, 1, 0], 
    #           [0, 0, 0, 0, 1, 1, 1], 
    #           [0, 0, 0, 0, 0, 1, 1], 
    #           [0, 0, 0, 0, 0, 0, 1], 
    #           [0, 0, 0, 0, 0, 0, 0]]
    # op_list = ['input', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'output']
    # model_spec = ModelSpec(matrix, op_list)
    # print(model_spec.matrix)
    # print(model_spec.ops)
    # indi = IndividualY()
    # indi.initialize()
    # print(indi)
    # indi.mutation()
    # print(indi)
