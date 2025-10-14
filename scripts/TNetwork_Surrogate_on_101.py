'''
Descripttion: 
version: 1.0
Author: ZXL
Date: 2024-05-07 10:58:17
LastEditors: ZXL
LastEditTime: 2025-10-14 17:16:30
'''
import os
import numpy as np
import random
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from evolve_on_101 import Evolution, query_fitness_for_indi, query_fitness_for_pops
from population import PopulationX
from individual import IndividualX
from make_sample import SampleSetX
from nasbench.lib import model_spec as _model_spec
from get_data_from_101 import NASBench101, padding_zero_in_matrix
from TNetwork_on_101 import TNetwork, TNetworkDataset, FusionLoss, train_TNetwork, test_TNetwork, category_center
from SiameseNetwork_on_101 import SiameseNetwork, SiameseNetworkDataset, train_SiameseNetwork, test_SiameseNetwork
from utils import population_log, write_best_individual, sampleset_log, get_current_time


ModelSpec = _model_spec.ModelSpec
nasbench101 = NASBench101()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Surrogate_Evolution(Evolution):
    def __init__(self, pc=0.2, pm=0.9, m_num_matrix=1, m_num_op_list=1, population_size=20):
        super().__init__(pc, pm, m_num_matrix, m_num_op_list, population_size)
        
        
    def set_pops(self, pops):
        self.pops.set_populations(pops)

    
    def copy_pops(self, pops: PopulationX):
        self.pops.copy_info_from_Population(pops)


    def pops2data(self, pops):
        data_list = []
        for indi in pops:
            matrix, op_list = indi.indi['matrix'], indi.indi['op_list']
            oper2feature = {'input': [1, 0, 0, 0, 0, 0], 'conv1x1-bn-relu': [0, 1, 0, 0, 0, 0], 'conv3x3-bn-relu': [0, 0, 1, 0, 0, 0], 
                        'maxpool3x3': [0, 0, 0, 1, 0, 0], 'null': [0, 0, 0, 0, 1, 0], 'output':[0, 0, 0, 0, 0, 1]}
            x = [oper2feature[oper] for oper in op_list]
            x = torch.tensor(x, dtype=torch.float)
            indices = np.where(matrix == 1)
            indices = np.array(indices)
            edge_index = torch.tensor(indices, dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)
            
        return data_list


    # def produce_archdata(self, recombinate_size):
    #     predict_sampleset = SampleSetX()
    #     offsprings = self.recombinate(recombinate_size)
    #     query_fitness_for_pops(gen_no, offsprings)
    #     predict_sampleset.add_arch_as_pops(offsprings)
    #     data_list = self.pops2data(offsprings.pops)
        
    #     return predict_sampleset, data_list


    def produce_archdata(self, recombinate_size):
        offsprings = self.recombinate(recombinate_size)
        query_fitness_for_pops(gen_no, offsprings)
        data_list = self.pops2data(offsprings.pops)
        
        return offsprings, data_list


    def produce_archdata_with_sample(self, recombinate_size, random_sample):
        predict_sampleset = SampleSetX()
        offsprings = self.recombinate(recombinate_size)
        query_fitness_for_pops(gen_no, offsprings)
        data_list = self.pops2data(offsprings.pops)
        predict_sampleset.add_arch_as_pops(offsprings)
        for _ in range(random_sample):
            index = random.randint(0, 423623)
            fixed_stat, _ = nasbench101.get_info_by_index(index)
            matrix, op_list = fixed_stat['module_adjacency'], fixed_stat['module_operations']
            padding_matrix, padding_op_list = padding_zero_in_matrix(matrix, op_list)
            indi = IndividualX()
            indi.create_an_individual(padding_matrix, padding_op_list)
            query_fitness_for_indi(indi)
            matrix, op_list = indi.indi['matrix'], indi.indi['op_list']
            oper2feature = {'input': [1, 0, 0, 0, 0, 0], 'conv1x1-bn-relu': [0, 1, 0, 0, 0, 0], 'conv3x3-bn-relu': [0, 0, 1, 0, 0, 0], 
                        'maxpool3x3': [0, 0, 0, 1, 0, 0], 'null': [0, 0, 0, 0, 1, 0], 'output':[0, 0, 0, 0, 0, 1]}
            x = [oper2feature[oper] for oper in op_list]
            x = torch.tensor(x, dtype=torch.float)
            indices = np.where(matrix == 1)
            indices = np.array(indices)
            edge_index = torch.tensor(indices, dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)
            predict_sampleset.add_arch_as_indi(indi)
        
        return predict_sampleset, data_list


    def predict_arch(self, model, dataset):
        embedding_list = []
        tier_list = []
        with torch.no_grad():
            for data in dataset:
                data = data.to(device)
                embedding, output = model(data.x, data.edge_index, data.batch)
                _, tier = torch.max(output, 1)
                embedding_list.append(embedding)
                tier_list.append(tier.item())
        
        return embedding_list, tier_list


    def environmental_selection_by_category_center(self, gen_no, train_sampleset, predict_sampleset, embedding_list, tier_list, elite_num, category_center, is_random=False):
        print('environmental selection...')
        higher_indices = [index for index, x in enumerate(tier_list) if x == tiers_num - 1]
        higher_num = len(higher_indices)
        if higher_num <= elite_num:
            original_pops = self.pops
            for indice in higher_indices:
                original_pops.append(predict_sampleset.archset[indice])
                train_sampleset.add_arch_as_indi(predict_sampleset.archset[indice])
            sorted_new_pops = sorted(original_pops, key=lambda x:x.mean_acc, reverse=True)
            new_pops = sorted_new_pops[:self.population_size]
            self.set_pops(new_pops)
        else:
            indices_similarity = {}
            category_center = category_center[tiers_num - 1]
            for indice in higher_indices:
                similarity = F.cosine_similarity(embedding_list[indice], category_center, dim=0)
                indices_similarity[indice] = similarity
            sorted_indices_similarity = dict(sorted(indices_similarity.item(), key= lambda item:item[1], reverse=True))
            last_indeces = list(sorted_indices_similarity.keys())[:5]
            original_pops = self.pops.pops
            for indice in last_indeces:
                original_pops.append(predict_sampleset.archset[indice])
                train_sampleset.add_arch_as_indi(predict_sampleset.archset[indice])
            sorted_new_pops = sorted(original_pops, key=lambda x:x.mean_acc, reverse=True)
            new_pops = sorted_new_pops[:self.population_size]
            self.set_pops(new_pops)
        population_log(gen_no, self.pops)
        write_best_individual(gen_no, self.pops)


    def environmental_selection_by_best_individual(self, TNet, gen_no, train_sampleset, predict_sampleset, embedding_list, tier_list, elite_num, category_center, is_random=False):
        print('environmental selection...')
        higher_indices = [index for index, x in enumerate(tier_list) if x == tiers_num - 1]
        higher_num = len(higher_indices)
        arg_index = self.pops.get_sorted_index_order_by_acc()
        best_individual = self.pops.get_individual(arg_index[0])
        data = arch2data(best_individual.indi['matrix'], best_individual.indi['op_list'])
        data = data.to(device)
        best_embedding, _ = TNet(data.x, data.edge_index, data.batch)
        if higher_num <= elite_num:
            original_pops = self.pops.pops
            candidate_indi = []
            for indice in higher_indices:
                candidate_indi.append(predict_sampleset.archset[indice])
                train_sampleset.add_arch_as_indi(predict_sampleset.archset[indice])
            sorted_original_pops = sorted(original_pops, key=lambda x:x.mean_acc, reverse=True)
            flag = 0
            for index in range(higher_num):
                if candidate_indi[index].mean_acc > sorted_original_pops[0].mean_acc or candidate_indi[index].mean_acc > sorted_original_pops[self.population_size-flag-1].mean_acc:
                    sorted_original_pops[self.population_size-flag-1] = candidate_indi[index]
                    flag += 1
            new_pops = sorted_original_pops
            self.set_pops(new_pops)
            for index in range(higher_num):
                print(candidate_indi[index])
        else:
            # indices_similarity = {}
            # for indice in higher_indices:
            #     similarity = F.cosine_similarity(embedding_list[indice], best_embedding, dim=1)
            #     indices_similarity[indice] = similarity.item()
            # sorted_indices_similarity = dict(sorted(indices_similarity.items(), key=lambda item:item[1], reverse=True))
            # last_indeces = list(sorted_indices_similarity.keys())[:elite_num]
            # original_pops = self.pops.pops
            # candidate_indi = []
            # for indice in last_indeces:
            #     candidate_indi.append(predict_sampleset.archset[indice])
            #     train_sampleset.add_arch_as_indi(predict_sampleset.archset[indice])
            # sorted_original_pops = sorted(original_pops, key=lambda x:x.mean_acc, reverse=True)
            # flag = 0
            # for index in range(elite_num):
            #     if candidate_indi[index].mean_acc > sorted_original_pops[0].mean_acc:
            #         sorted_original_pops[self.population_size-flag-1] = candidate_indi[index]
            #         flag += 1
            # new_pops = sorted_original_pops
            # self.set_pops(new_pops)
            original_pops = self.pops.pops
            index_list = random.sample(range(0, higher_num), elite_num)
            candidate_indi = []
            for index in index_list:
                candidate_indi.append(predict_sampleset.archset[higher_indices[index]])
                train_sampleset.add_arch_as_indi(predict_sampleset.archset[higher_indices[index]])
            sorted_original_pops = sorted(original_pops, key=lambda x:x.mean_acc, reverse=True)
            flag = 0
            for index in range(elite_num):
                if candidate_indi[index].mean_acc > sorted_original_pops[0].mean_acc or candidate_indi[index].mean_acc > sorted_original_pops[self.population_size-flag-1].mean_acc:
                    sorted_original_pops[self.population_size-flag-1] = candidate_indi[index]
                    flag += 1
            new_pops = sorted_original_pops
            self.set_pops(new_pops)
            print(candidate_indi[0])
            print(candidate_indi[1])
            print(candidate_indi[2])
        population_log(gen_no, self.pops)
        write_best_individual(gen_no, self.pops)

    
    def environmental_selection_by_SNet(self, TNet, SNet, gen_no, train_sampleset, offsprings, embedding_list, tier_list, elite_num, category_center, is_random=False):
        print('environmental selection...')
        higher_indices = [index for index, x in enumerate(tier_list) if x == tiers_num - 1]
        max = 0
        for index in higher_indices:
            if offsprings.pops[index].mean_acc > max:
                max = offsprings.pops[index].mean_acc
        print(max)
        higher_num = len(higher_indices)
        arg_index = self.pops.get_sorted_index_order_by_acc()
        best_individual = self.pops.get_individual(arg_index[0])
        data = arch2data(best_individual.indi['matrix'], best_individual.indi['op_list'])
        data = data.to(device)
        anchor = data
        anchor_indi = best_individual
        if higher_num == 0:
            candidate_indi = []
            index_list = random.sample(range(0, offsprings.pops_size), elite_num)
            for index in index_list:
                candidate_indi.append(offsprings.pops[index])
            offspring_pops = PopulationX(0)
            offspring_pops.set_populations(candidate_indi)
            self.environmental_selection(gen_no, offspring_pops)
            for index in range(elite_num):
                print(candidate_indi[index])
        elif higher_num <= elite_num:
            candidate_indi = []
            for indice in higher_indices:
                candidate_indi.append(offsprings.pops[indice])
                train_sampleset.add_arch_as_indi(offsprings.pops[indice])
            offspring_pops = PopulationX(0)
            offspring_pops.set_populations(candidate_indi)
            self.environmental_selection(gen_no, offspring_pops)
            for index in range(higher_num):
                print(candidate_indi[index])
        else:
            candidate_indi = []
            for indice in higher_indices:
                candidate_indi.append(offsprings.pops[indice])
            query_indi = []
            flag = False
            for indi in candidate_indi:
                data = arch2data(indi.indi['matrix'], indi.indi['op_list'])
                data.to(device)
                output = SNet(data, anchor)
                _, predicted = torch.max(output, dim=1)
                if predicted == 1:
                    anchor = data
                    anchor_indi = indi
                    flag = True
                    print(anchor_indi)

            if flag:
                query_indi.append(anchor_indi)
                index_list = random.sample(range(0, higher_num), elite_num-1)
                for index in index_list:
                    query_indi.append(candidate_indi[index])
            else:
                index_list = random.sample(range(0, higher_num), elite_num)
                for index in index_list:
                    query_indi.append(candidate_indi[index])
            offspring_pops = PopulationX(0)
            offspring_pops.set_populations(candidate_indi)
            self.environmental_selection(gen_no, offspring_pops)


def update_surrogate_model(gen_no, sampleset, EPOCH, tiers_num):
    data_sampled = sampleset.get_data_sampled()
    dataset = TNetworkDataset(data_sampled=data_sampled, tiers_num=tiers_num, augmentation=True)
    train_loder = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    TNet = TNetwork(input_channels=dataset.num_node_features, hidden_channels=64, output_channels=dataset.num_classes).to(device)
    optimizer = optim.Adam(TNet.parameters())
    criterion = FusionLoss(0.6)
    train_TNetwork(TNet, EPOCH, train_loder, optimizer, criterion)
    save_path = r'./predictor/PT/TNet_{}.pt'.format(gen_no)
    torch.save(TNet.state_dict(), save_path)
    center = category_center(TNet, train_loder, tiers_num)

    return TNet, center


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




if __name__ == '__main__':
    # hyperparametes
    pc = 0.8
    pm = 0.2
    m_num_matrix = 1
    m_num_op_list = 1
    population_size = 50
    recombinate_size = 2000
    random_sample = 9000
    generation_num = 3
    surrogate_num = 47
    update_frequency = 20

    tiers_num = 5
    elite_num = 2
    EPOCH = 50
    batch_size = 64

    Evolution = Evolution(pc, pm, m_num_matrix, m_num_op_list, population_size)
    train_sampleset = SampleSetX()
    Evolution.initialize_popualtion()
    
    gen_no = 0
    query_fitness_for_pops(gen_no, Evolution.pops)
    train_sampleset.add_arch_as_pops(Evolution.pops)
    print('general evolutionary process')
    while True:
        gen_no += 1
        if gen_no > generation_num:
            break
        print('{}th generation:'.format(gen_no))
        offsprings = Evolution.recombinate(population_size)
        query_fitness_for_pops(gen_no, offsprings)
        train_sampleset.add_arch_as_pops(offsprings)
        Evolution.environmental_selection(gen_no, offsprings)
    gen_no -= 1
    # sampleset_log(train_sampleset)
    # current_time = get_current_time()
    # save_path = r'./pkl/train_sampleset_{}.pkl'.format(current_time)
    # with open(save_path, 'wb') as file:
    #     pickle.dump(train_sampleset, file)
    
    print('start training surrsogate model...')
    data_sampled = train_sampleset.get_data_sampled()
    dataset = TNetworkDataset(data_sampled=data_sampled, tiers_num=tiers_num, augmentation=True)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    TNet = TNetwork(input_channels=dataset.num_node_features, hidden_channels=64, output_channels=dataset.num_classes).to(device)
    optimizer = optim.Adam(TNet.parameters())
    criterion = FusionLoss(0.6)
    # train_TNetwork(TNet, EPOCH, train_loader, optimizer, criterion)
    # save_path = r'./predictor/PT/TNet_{}.pt'.format(gen_no)
    # torch.save(TNet.state_dict(), save_path)
    # print('complete surrogate model training...')
    
    save_path = r'./predictor/Queries Number/200_TNetwork-F_2024-07-05_09-51-49.pt'
    TNet.load_state_dict(torch.load(save_path))
    TNet.eval()

    SNet = SiameseNetwork(6, 64, 2).to(device)
    save_path = r'./predictor/Queries Number/200_SiameseNetwork_2024-07-05_09-57-15.pt'
    SNet.load_state_dict(torch.load(save_path))
    SNet.eval()

    print('surrogate_assisted evolutionary process')
    Surrogate_Evolution = Surrogate_Evolution(pc, pm, m_num_matrix, m_num_op_list, population_size)
    Surrogate_Evolution.copy_pops(Evolution.pops)
    center = category_center(TNet, train_loader, tiers_num)
    while True:
        gen_no += 1
        if gen_no > generation_num + surrogate_num:
            break
        print('{}th generation:'.format(gen_no))
        # if gen_no == 31:
        #     TNet, center = update_surrogate_model(gen_no, train_sampleset, EPOCH, tiers_num)
        offsprings, data_list = Surrogate_Evolution.produce_archdata(recombinate_size)
        embedding_list, tier_list = Surrogate_Evolution.predict_arch(TNet, data_list)
        Surrogate_Evolution.environmental_selection_by_SNet(TNet, SNet, gen_no, train_sampleset, offsprings, embedding_list, tier_list, elite_num, center)
    
    arg_index = Surrogate_Evolution.pops.get_sorted_index_order_by_acc()
    best_individual = Surrogate_Evolution.pops.get_individual(arg_index[0])
    print('Global optimal solution:')
    print(best_individual)
    model_spec = ModelSpec(matrix=best_individual.indi['matrix'], ops=best_individual.indi['op_list'])
    print(nasbench101.get_info_by_model_spec(model_spec))