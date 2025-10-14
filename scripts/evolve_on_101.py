'''
Descripttion: 
version: 1.0
Author: ZXL
Date: 2024-01-23 11:13:42
LastEditors: ZXL
LastEditTime: 2025-10-14 17:05:36
'''
import os
import pickle
import copy
import numpy as np
import random
import time
from individual import IndividualX
from population import PopulationX
from make_sample import SampleSetX
from utils import get_newseed, seed_log, utl2matrix, matrix2utl, population_log, write_best_individual, find_all_simple_paths, build_adj_matrix_from_paths
from nasbench.lib import model_spec as _model_spec
from get_data_from_101 import NASBench101


ModelSpec = _model_spec.ModelSpec
nasbench101 = NASBench101()


def query_fitness_for_indi(query_indi:IndividualX):
    # print('query fitness for individual...')
    model_spec = ModelSpec(matrix=query_indi.indi['matrix'], ops=query_indi.indi['op_list'])
    _, computed_stat = nasbench101.get_info_by_model_spec(model_spec)
    final_valid_accuracy_list = []
    final_test_accuracy_list = []
    x = random.randint(0, 2)
    random_final_valid_accuracy = computed_stat[108][x]['final_validation_accuracy']
    for i in range(3):
        final_valid_accuracy_list.append(computed_stat[108][i]['final_validation_accuracy'])
        final_test_accuracy_list.append(computed_stat[108][i]['final_test_accuracy'])
    mean_final_valid_accuracy = np.mean(final_valid_accuracy_list)
    mean_final_test_accuracy = np.mean(final_test_accuracy_list)

    query_indi.mean_acc = mean_final_valid_accuracy
    query_indi.random_acc = random_final_valid_accuracy
    query_indi.test_mean_acc = mean_final_test_accuracy
    

def query_fitness_for_pops(gen_no, query_pop: PopulationX):
    # print('query fitness for population {}'.format(gen_no))
    for i, indi in enumerate(query_pop.pops):
        model_spec = ModelSpec(matrix=indi.indi['matrix'], ops=indi.indi['op_list'])
        _, computed_stat = nasbench101.get_info_by_model_spec(model_spec)
        final_valid_accuracy_list = []
        final_test_accuracy_list = []
        x = random.randint(0, 2)
        random_final_valid_accuracy = computed_stat[108][x]['final_validation_accuracy']
        for j in range(3):
            final_valid_accuracy_list.append(computed_stat[108][j]['final_validation_accuracy'])
            final_test_accuracy_list.append(computed_stat[108][j]['final_test_accuracy'])
        mean_final_valid_accuracy = np.mean(final_valid_accuracy_list)
        mean_final_test_accuracy = np.mean(final_test_accuracy_list)

        query_pop.pops[i].mean_acc = mean_final_valid_accuracy
        query_pop.pops[i].random_acc = random_final_valid_accuracy
        query_pop.pops[i].test_mean_acc = mean_final_test_accuracy
    population_log(gen_no, query_pop)


class Evolution():
    def __init__(self, pc=0.2, pm=0.9, m_num_matrix=1, m_num_op_list=1, population_size=20):
        self.pc = pc
        self.pm = pm
        self.m_num_matrix = m_num_matrix
        self.m_num_op_list = m_num_op_list
        self.population_size = population_size
        self.pops = PopulationX(population_size)

    def initialize_popualtion(self):
        print("initializing population with number {}...".format(self.population_size))
        init_path = r'./pkl/init_population_{}.pkl'.format(self.population_size)
        # if os.path.exists(init_path):
        #     # load population
        #     print('loading population...')
        #     with open(init_path, 'rb') as file:
        #         self.pops = pickle.load(file)
        # else:
        print('generate population...')
        self.pops = PopulationX(self.population_size, self.m_num_matrix, self.m_num_op_list)
        with open(init_path, 'wb') as file:
            pickle.dump(self.pops, file)
        # all the initialized population should be saved
        population_log(0, self.pops)
        

    def recombinate(self, pop_size) -> PopulationX:
        print('mutation and crossover...')
        offspring_list = []
        for _ in range(int(pop_size / 2)):
            # tournament selection
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
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
        offspring_pops = PopulationX(0)
        offspring_pops.set_populations(offspring_list)

        return offspring_pops
    

    # def crossover(self, p1: IndividualX, p2: IndividualX, utl_len=21, op_list_len=7):
    #     p1 = copy.deepcopy(p1)
    #     p2 = copy.deepcopy(p2)
    #     p1.clear_state_info()
    #     p2.clear_state_info()
    #     utl1 = matrix2utl(p1.indi['matrix'])
    #     utl2 = matrix2utl(p2.indi['matrix'])
    #     retry_num = 0
    #     while True:
    #         retry_num += 1
    #         cross_point = random.randint(1, utl_len - 1)
    #         crossed_utl1 = np.hstack((utl1[:cross_point], utl2[cross_point:]))
    #         crossed_utl2 = np.hstack((utl2[:cross_point], utl1[cross_point:]))
    #         crossed_matrix1 = utl2matrix(crossed_utl1)
    #         crossed_matrix2 = utl2matrix(crossed_utl2)
    #         model_spec1 = ModelSpec(matrix=crossed_matrix1, ops=p1.indi['op_list'])
    #         model_spec2 = ModelSpec(matrix=crossed_matrix2, ops=p2.indi['op_list'])
    #         # considering the invalid spec
    #         if model_spec1.valid_spec and (np.sum(model_spec1.matrix) <= 9) and model_spec2.valid_spec and (
    #             np.sum(model_spec2.matrix) <= 9):
    #             break
    #         if retry_num > 20:
    #             print('Crossover has tried for more than 20 times, but still get invalid spec.\n'
    #                   'Give up this crossover and go on...')
    #             crossed_matrix1 = p1.indi['matrix']
    #             crossed_matrix2 = p2.indi['matrix']
    #             break
    #     p1.indi['matrix'] = crossed_matrix1
    #     p2.indi['matrix'] = crossed_matrix2
    #     op_list1 = p1.indi['op_list']
    #     op_list2 = p2.indi['op_list']
    #     op_list_cross_point = random.randint(1, op_list_len - 1)
    #     crossed_op_list1 = np.hstack((op_list1[:op_list_cross_point], op_list2[op_list_cross_point:]))
    #     crossed_op_list2 = np.hstack((op_list2[:op_list_cross_point], op_list1[op_list_cross_point:]))
    #     p1.indi['op_list'] = crossed_op_list1.tolist()
    #     p2.indi['op_list'] = crossed_op_list2.tolist()
            
    #     return p1, p2
    

    def crossover(self, p1: IndividualX, p2: IndividualX):
        p1 = copy.deepcopy(p1)
        p2 = copy.deepcopy(p2)
        p1.clear_state_info()
        p2.clear_state_info()
        retry_num = 0
        while True:
            retry_num += 1
            paths1 = []
            paths2 = []
            op_list1 = []
            op_list2 = []
            paths1 = find_all_simple_paths(p1.indi['matrix'], 0, 6)
            paths2 = find_all_simple_paths(p2.indi['matrix'], 0, 6)
            paths = paths1 + paths2
            if retry_num > 20 or len(paths) == 0:
                # print('Crossover has tried for more than 20 times, but still get invalid spec.\n'
                #         'Give up this crossover and go on...')
                op_list1 = p1.indi['op_list']
                op_list2 = p2.indi['op_list']
                op_list_cross_point = random.randint(1, 6)
                crossed_op_list1 = np.hstack((op_list1[:op_list_cross_point], op_list2[op_list_cross_point:]))
                crossed_op_list2 = np.hstack((op_list2[:op_list_cross_point], op_list1[op_list_cross_point:]))
                p1.indi['op_list'] = crossed_op_list1.tolist()
                p2.indi['op_list'] = crossed_op_list2.tolist()
                break
            selected_index1 = random.sample(range(0, len(paths)), int(len(paths) / 2) + 1)
            selected_index2 = random.sample(range(0, len(paths)), int(len(paths) / 2) + 1)
            for index in selected_index1:
                paths1.append(paths[index])
            for index in selected_index2:
                paths2.append(paths[index])
            crossed_matrix1 = build_adj_matrix_from_paths(paths1, 7)
            crossed_matrix2 = build_adj_matrix_from_paths(paths2, 7)
            op_list1 = p1.indi['op_list']
            op_list2 = p2.indi['op_list']
            op_list_cross_point = random.randint(1, 6)
            crossed_op_list1 = np.hstack((op_list1[:op_list_cross_point], op_list2[op_list_cross_point:]))
            crossed_op_list2 = np.hstack((op_list2[:op_list_cross_point], op_list1[op_list_cross_point:]))
            crossed_op_list1, crossed_op_list2 = crossed_op_list1.tolist(), crossed_op_list2.tolist()

            model_spec1 = ModelSpec(matrix=crossed_matrix1, ops=crossed_op_list1)
            model_spec2 = ModelSpec(matrix=crossed_matrix2, ops=crossed_op_list2)
            # considering the invalid spec
            if model_spec1.valid_spec and (np.sum(model_spec1.matrix) <= 9) and model_spec2.valid_spec and (
                np.sum(model_spec2.matrix) <= 9):
                p1.indi['matrix'], p1.indi['op_list'] = crossed_matrix1, crossed_op_list1
                p2.indi['matrix'], p2.indi['op_list'] = crossed_matrix2, crossed_op_list2
                break
        
        return p1, p2

    def environmental_selection(self, gen_no, offspring_population: PopulationX, is_random=False):
        # environment selection from the current population and the offspring population
        # assert (self.pops.get_pop_size() == self.population_size)
        # assert (offspring_population.get_pop_size() == self.population_size)
        print('environmental selection...')
        elitism = 0.05
        elite_num = int(self.population_size * elitism)
        indi_list = self.pops.pops
        indi_list.extend(offspring_population.pops)
        # descending order
        if is_random:
            indi_list.sort(key=lambda x: x.random_acc, reverse=True)
        indi_list.sort(key=lambda x: x.mean_acc, reverse=True)
        elitism_list = indi_list[0:elite_num]
        left_list = indi_list[elite_num:]
        np.random.shuffle(left_list)
        
        num = 0
        while(num < self.population_size - elite_num):
            flag = True
            i1 = random.randint(0, len(left_list) - 1)
            i2 = random.randint(0, len(left_list) - 1)
            winner = self.selection(left_list[i1], left_list[i2], is_random)
            for indi in elitism_list:
                if np.array_equal(indi.indi['matrix'], winner.indi['matrix']) and indi.indi['op_list'] == winner.indi['op_list']:
                    flag = False
                    break
            if flag:
                elitism_list.append(winner)
                num += 1
        # for _ in range(self.population_size - elite_num):
        #     i1 = random.randint(0, len(left_list) - 1)
        #     i2 = random.randint(0, len(left_list) - 1)
        #     winner = self.selection(left_list[i1], left_list[i2], is_random)
            
        #     elitism_list.append(winner)

        self.pops.set_populations(elitism_list)
        # record each generation's population and best individual
        population_log(gen_no, self.pops)
        write_best_individual(gen_no, self.pops)


    def tournament_selection(self):
        ind1_id = random.randint(0, self.pops.get_pop_size() - 1)
        ind2_id = random.randint(0, self.pops.get_pop_size() - 1)
        ind1 = self.pops.get_individual(ind1_id)
        ind2 = self.pops.get_individual(ind2_id)
        winner = self.selection(ind1, ind2)

        return winner
    

    def selection(self, ind1, ind2, is_random=False):
        if is_random:
            if ind1.random_acc > ind2.random_acc:
                return ind1
            else:
                return ind2
        if ind1.mean_acc > ind2.mean_acc:
            return ind1
        else:
            return ind2


    def set_probability(self):
        prob = []
        for i in range(self.population_size):
            indi = self.pops.pops[i]
            prob.append(indi.mean_acc)
        total = sum(prob)
        prob = prob / total

        return prob


    # roulette wheel selection
    def roulette_selection(self, prob):
        times = int(self.population_size / 2)
        start_list = [random.randint(0, self.population_size) for _ in range(self.population_size)]
        pointer_list = [random.random() for _ in range(times)]
        select = np.zeros(times)
        for i in range(times):
            sumP = 0
            start = start_list[i]
            pointer = pointer_list[i]
            while sumP < pointer:
                sumP += prob[start % self.population_size + 1]
                start += 1
            select[i] = (start - 1) % self.population_size

        elements, counts = np.unique(select, return_counts=True)
        most_element_index = np.argmax(counts)
        most_element = elements[most_element_index]

        return most_element




if __name__ == '__main__':
    # current_seed = get_newseed()
    # # seed_log(current_seed)
    # np.random.seed(current_seed)
    # hyperparametes
    pc = 0.8
    pm = 0.8
    m_num_matrix = 1
    m_num_op_list = 1
    population_size = 100
    generation_num = 50
    Evolution = Evolution(pc, pm, m_num_matrix, m_num_op_list, population_size)
    train_sampleset = SampleSetX()
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

    # save_path = r'./pkl/sampleset_{}.pkl'.format(200)
    # with open(save_path, 'wb') as file:
    #     pickle.dump(train_sampleset, file)