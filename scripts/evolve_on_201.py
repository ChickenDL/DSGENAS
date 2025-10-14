'''
Descripttion: 
version: 1.0
Author: ZXL
Date: 2024-01-23 11:13:42
LastEditors: ZXL
LastEditTime: 2025-10-14 17:06:27
'''
import os
import pickle
import copy
from get_data_from_201 import NASBench201
import numpy as np
import random
from individual import IndividualY
from population import PopulationY
from make_sample import SampleSetY
from utils import get_newseed, seed_log, utl2matrix, matrix2utl, population_log, write_best_individual
from nasbench.lib import model_spec as _model_spec


ModelSpec = _model_spec.ModelSpec
nasbench201 = NASBench201()

def op_list2str(op_list):
    op_str = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(op_list[0], op_list[1], op_list[2], op_list[3], op_list[4],
                                                          op_list[5])
    return op_str


def modify_matrix(indi:IndividualY):
    matrix = np.array([[0, 1, 1, 1], 
                       [0, 0, 1, 1],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0]], dtype='int8')
    for i in range(6):
        if indi.indi['op_list'][i] == 'none':
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
    indi.indi['matrix'] = matrix


def query_fitness_for_indi(query_indi:IndividualY):
    # print('query fitness for individual...')
    op_list = query_indi.indi['op_list']
    arch_str = op_list2str(op_list)
    info = nasbench201.get_info_by_arch_str(arch_str)
    query_indi.mean_acc['cifar10_valid'] = info['cifar10_valid']
    query_indi.mean_acc['cifar10_test'] = info['cifar10_test']
    query_indi.mean_acc['cifar100_valid'] = info['cifar100_valid']
    query_indi.mean_acc['cifar100_test'] = info['cifar100_test']
    query_indi.mean_acc['ImageNet_valid'] = info['ImageNet_valid']
    query_indi.mean_acc['ImageNet_test'] = info['ImageNet_test']
    

def query_fitness_for_pops(gen_no, query_pop: PopulationY):
    # print('query fitness for population {}'.format(gen_no))
    for i, indi in enumerate(query_pop.pops):
        op_list = indi.indi['op_list']
        arch_str = op_list2str(op_list)
        info = nasbench201.get_info_by_arch_str(arch_str)
        query_pop.pops[i].mean_acc['cifar10_valid'] = info['cifar10_valid']
        query_pop.pops[i].mean_acc['cifar10_test'] = info['cifar10_test']
        query_pop.pops[i].mean_acc['cifar100_valid'] = info['cifar100_valid']
        query_pop.pops[i].mean_acc['cifar100_test'] = info['cifar100_test']
        query_pop.pops[i].mean_acc['ImageNet_valid'] = info['ImageNet_valid']
        query_pop.pops[i].mean_acc['ImageNet_test'] = info['ImageNet_test']

    population_log(gen_no, query_pop)


class Evolution():
    def __init__(self, pc=0.2, pm=0.9, m_num_matrix=1, m_num_op_list=1, population_size=20, dataset='cifar10_valid'):
        self.pc = pc
        self.pm = pm
        self.m_num_matrix = m_num_matrix
        self.m_num_op_list = m_num_op_list
        self.population_size = population_size
        self.dataset = dataset
        self.pops = PopulationY(population_size)


    def initialize_popualtion(self):
        print("initializing population with number {}...".format(self.population_size))
        init_path = r'./pkl/init_population_{}.pkl'.format(self.population_size)
        if os.path.exists(init_path):
            # load population
            print('loading population...')
            with open(init_path, 'rb') as file:
                self.pops = pickle.load(file)
        else:
            print('generate population...')
            self.pops = PopulationY(self.population_size, self.m_num_matrix, self.m_num_op_list)
            with open(init_path, 'wb') as file:
                pickle.dump(self.pops, file)
        # all the initialized population should be saved
        population_log(0, self.pops)
        

    def recombinate(self, pop_size) -> PopulationY:
        print('mutation and crossover...')
        offspring_list = []
        for _ in range(int(pop_size / 2)):
            # tournament selection
            p1 = self.tournament_selection(self.dataset)
            p2 = self.tournament_selection(self.dataset)
            # crossover
            if random.random() < self.pc:
                offset1, offset2 = self.crossover(p1, p2)
            else:
                offset1 = copy.deepcopy(p1)
                offset2 = copy.deepcopy(p2)
            # mutation
            if random.random() < self.pm:
                offset1.mutation()
            if random.random() < self.pm:
                offset2.mutation()
            offspring_list.append(offset1)
            offspring_list.append(offset2)
        offspring_pops = PopulationY(0)
        offspring_pops.set_populations(offspring_list)

        return offspring_pops
    

    def crossover(self, p1: IndividualY, p2: IndividualY, utl_len=21, op_list_len=5):
        p1 = copy.deepcopy(p1)
        p2 = copy.deepcopy(p2)
        p1.clear_state_info()
        p2.clear_state_info()
        op_list1 = p1.indi['op_list']
        op_list2 = p2.indi['op_list']
        op_list_cross_point = random.randint(1, op_list_len - 1)
        crossed_op_list1 = np.hstack((op_list1[:op_list_cross_point], op_list2[op_list_cross_point:]))
        crossed_op_list2 = np.hstack((op_list2[:op_list_cross_point], op_list1[op_list_cross_point:]))
        p1.indi['op_list'] = crossed_op_list1.tolist()
        p2.indi['op_list'] = crossed_op_list2.tolist()
        modify_matrix(p1)
        modify_matrix(p2)
            
        return p1, p2


    def environmental_selection(self, gen_no, offspring_population: PopulationY):
        # environment selection from the current population and the offspring population
        # assert (self.pops.get_pop_size() == self.population_size)
        # assert (offspring_population.get_pop_size() == self.population_size)
        print('environmental selection...')
        elitism = 0.2
        elite_num = int(self.population_size * elitism)
        indi_list = self.pops.pops
        indi_list.extend(offspring_population.pops)
        # descending order
        indi_list.sort(key=lambda x: x.mean_acc[self.dataset], reverse=True)
        elitism_list = indi_list[0:elite_num]
        left_list = indi_list[elite_num:]
        np.random.shuffle(left_list)
        
        for _ in range(self.population_size - elite_num):
            i1 = random.randint(0, len(left_list) - 1)
            i2 = random.randint(0, len(left_list) - 1)
            winner = self.selection(left_list[i1], left_list[i2], self.dataset)
            elitism_list.append(winner)

        self.pops.set_populations(elitism_list)
        # record each generation's population and best individual
        population_log(gen_no, self.pops)
        arg_index = self.pops.get_sorted_index_order_by_acc(self.dataset)
        best_individual = self.pops.get_individual(arg_index[0])
        save_path = r'./output/pops_log/best_acc.txt'
        with open(save_path, 'a') as myfile:
            myfile.write('gen_no: {}'.format(gen_no) + '\n')
            myfile.write(str(best_individual))
            myfile.write('\n')


    def tournament_selection(self, dataset):
        ind1_id = random.randint(0, self.pops.get_pop_size() - 1)
        ind2_id = random.randint(0, self.pops.get_pop_size() - 1)
        ind1 = self.pops.get_individual(ind1_id)
        ind2 = self.pops.get_individual(ind2_id)
        winner = self.selection(ind1, ind2, dataset)

        return winner
    

    def selection(self, ind1, ind2, dataset):
        if ind1.mean_acc[dataset] > ind2.mean_acc[dataset]:
            return ind1
        else:
            return ind2
    



if __name__ == '__main__':
    current_seed = get_newseed()
    # seed_log(current_seed)
    np.random.seed(current_seed)
    # hyperparametes
    pc = 0.8
    pm = 0.2
    m_num_matrix = 1
    m_num_op_list = 1
    population_size = 200
    generation_num = 1
    dataset = 'ImageNet_valid'
    Evolution = Evolution(pc, pm, m_num_matrix, m_num_op_list, population_size, dataset)
    train_sampleset = SampleSetY()
    # Evolution.initialize_popualtion()
    gen_no = 0
    query_fitness_for_pops(gen_no, Evolution.pops)
    train_sampleset.add_arch_as_pops(Evolution.pops)
    while True:
        gen_no += 1
        if gen_no > generation_num:
            break
        print('{}th generation:'.format(gen_no))
        offsprings = Evolution.recombinate(population_size)
        query_fitness_for_pops(gen_no, offsprings)
        train_sampleset.add_arch_as_pops(offsprings)
        Evolution.environmental_selection(gen_no, offsprings)

    save_path = r'./pkl/201_sampleset_{}_{}.pkl'.format(200, dataset)
    with open(save_path, 'wb') as file:
        pickle.dump(train_sampleset, file)
    