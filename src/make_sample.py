'''
Descripttion: 
version: 1.0
Author: ZXL
Date: 2024-01-31 20:50:08
LastEditors: ZXL
LastEditTime: 2024-05-13 14:44:07
'''
from __future__ import absolute_import
from scheme.nodeinfo_transmit import transmits
from individual import IndividualX, IndividualY
from population import PopulationX, PopulationY
from nasbench.lib import model_spec as _model_spec
from get_data_from_101 import padding_zero_in_matrix


ModelSpec = _model_spec.ModelSpec

class SampleSetX():
    def __init__(self):
        self.archset = []
        self.archsize = 0

    
    def add_arch_as_pops(self, pops):
        for indi in pops.pops:
            self.archset.append(indi)
        self.archsize += pops.pops_size
    

    def add_arch_as_indi(self, indi):
        self.archset.append(indi)
        self.archsize += 1


    def get_data_sampled(self):
        data_sampled = []
        for arch in self.archset:
            new_sampled = {}
            new_sampled_arch = {}
            matrix, op_list = arch.indi['matrix'], arch.indi['op_list']
            new_sampled_arch['matrix'] = matrix
            new_sampled_arch['op_list'] = op_list
            new_sampled_acc = arch.mean_acc
            new_sampled['arch'] = new_sampled_arch
            new_sampled['acc'] = new_sampled_acc
            data_sampled.append(new_sampled)

        return data_sampled
            

    def set_label(self, tiers_num):
        labels = []
        # calculate the quantity of each tier
        tier_size = int(self.archsize / tiers_num)
        # sort
        sorted_archset = sorted(self.archset, key=lambda x:x.mean_acc, reverse=True)
        # tiers based on quantity
        for tier in range(tiers_num):
            start_index = tier * tier_size
            end_index = min((tier + 1) * tier_size, self.archsize)
            labels[start_index:end_index] = [tier] * (end_index - start_index)
        return sorted_archset, labels


    def get_surroggate_dataset(self, tiers_num):
        sorted_archset, labels = self.set_label(tiers_num)
        features = []
        # encode
        for arch in sorted_archset:
            matrix, op_list = arch.indi['matrix'], arch.indi['op_list']
            model_spec = ModelSpec(matrix, op_list)
            pruned_matrix, pruned_op_list = model_spec.matrix, model_spec.ops
            padding_matrix, padding_op_list = padding_zero_in_matrix(pruned_matrix, pruned_op_list)
            transmit_matrix = transmits(padding_matrix, padding_op_list)
            transmit_vector = transmit_matrix.flatten()
            features.append(transmit_vector)
            self.arch2encode[arch] = transmit_vector
        return features, labels


    def __str__(self):
        str_ = []
        for i in range(self.archsize):
            str_.append(str(self.archset[i]))
        return '\n'.join(str_)


class SampleSetY():
    def __init__(self):
        self.archset = []
        self.archsize = 0

    
    def add_arch_as_pops(self, pops):
        for indi in pops.pops:
            self.archset.append(indi)
        self.archsize += pops.pops_size
    

    def add_arch_as_indi(self, indi):
        self.archset.append(indi)
        self.archsize += 1


    def get_data_sampled(self, dataset):
        data_sampled = []
        for arch in self.archset:
            new_sampled = {}
            new_sampled_arch = {}
            matrix, op_list = arch.indi['matrix'], arch.indi['op_list']
            new_sampled_arch['matrix'] = matrix
            new_sampled_arch['op_list'] = op_list
            new_sampled_acc = arch.mean_acc[dataset]
            new_sampled['arch'] = new_sampled_arch
            new_sampled['acc'] = new_sampled_acc
            data_sampled.append(new_sampled)

        return data_sampled


    def __str__(self):
        str_ = []
        for i in range(self.archsize):
            str_.append(str(self.archset[i]))
        return '\n'.join(str_)
    

class SampleSet():
    def __init__(self):
        self.archset = []
        self.archsize = 0

    
    def add_arch_as_pops(self, pops):
        for indi in pops.pops:
            self.archset.append(indi)
        self.archsize += pops.pops_size
    

    def add_arch_as_indi(self, indi):
        self.archset.append(indi)
        self.archsize += 1


    def get_data_sampled(self):
        data_sampled = []
        for arch in self.archset:
            new_sampled = {}
            new_sampled_arch = {}
            matrix, op_list = arch.indi['matrix'], arch.indi['op_list']
            new_sampled_arch['matrix'] = matrix
            new_sampled_arch['op_list'] = op_list
            new_sampled_acc = arch.mean_acc
            new_sampled['arch'] = new_sampled_arch
            new_sampled['acc'] = new_sampled_acc
            data_sampled.append(new_sampled)

        return data_sampled
            

    def set_label(self, tiers_num):
        labels = []
        # calculate the quantity of each tier
        tier_size = int(self.archsize / tiers_num)
        # sort
        sorted_archset = sorted(self.archset, key=lambda x:x.mean_acc, reverse=True)
        # tiers based on quantity
        for tier in range(tiers_num):
            start_index = tier * tier_size
            end_index = min((tier + 1) * tier_size, self.archsize)
            labels[start_index:end_index] = [tier] * (end_index - start_index)
        return sorted_archset, labels


    def get_surroggate_dataset(self, tiers_num):
        sorted_archset, labels = self.set_label(tiers_num)
        features = []
        # encode
        for arch in sorted_archset:
            matrix, op_list = arch.indi['matrix'], arch.indi['op_list']
            model_spec = ModelSpec(matrix, op_list)
            pruned_matrix, pruned_op_list = model_spec.matrix, model_spec.ops
            padding_matrix, padding_op_list = padding_zero_in_matrix(pruned_matrix, pruned_op_list)
            transmit_matrix = transmits(padding_matrix, padding_op_list)
            transmit_vector = transmit_matrix.flatten()
            features.append(transmit_vector)
            self.arch2encode[arch] = transmit_vector
        return features, labels


    def __str__(self):
        str_ = []
        for i in range(self.archsize):
            str_.append(str(self.archset[i]))
        return '\n'.join(str_)




if __name__ == '__main__':
    sampleset = SampleSetX()
    indi = IndividualX()
    indi.initialize()
    pop = PopulationX(pops_size=9, m_num_matrix=2, m_num_op_list=2)
    sampleset.add_arch_as_indi(indi)
    sampleset.add_arch_as_pops(pop)
    # print(sampleset.archsize)
    # print(str(sampleset))
    # print(sampleset.get_surroggate_dataset_by_transmit(5))
    data_sampled = sampleset.get_data_sampled()
    print(data_sampled)