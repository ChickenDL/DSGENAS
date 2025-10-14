from individual import IndividualX, IndividualY
import numpy as np


class PopulationX:
    def __init__(self, pops_size, m_num_matrix=1, m_num_op_list=1):
        self.pops_size = pops_size
        self.pops = []
        for i in range(pops_size):
            indi = IndividualX(m_num_matrix, m_num_op_list)
            indi.initialize()
            self.pops.append(indi)

    
    def copy_info_from_Population(self, copy_population):
        assert len(self.pops) == len(copy_population.pops)
        for i, indi in enumerate(copy_population.pops):
            self.pops[i].indi['matrix'] = indi.indi['matrix']
            self.pops[i].indi['op_list'] = indi.indi['op_list']
            self.pops[i].mean_acc = indi.mean_acc
            self.pops[i].random_acc = indi.random_acc
            self.pops[i].test_mean_acc = indi.test_mean_acc

    
    def get_individual(self, i):
        return self.pops[i]
    
    
    def get_pop_size(self):
        return len(self.pops)
    

    def set_populations(self, new_pops):
        self.pops = new_pops
        self.pops_size = len(new_pops)


    def merge_populations(self, new_pops):
        for indi in new_pops:
            self.pops.append(indi)
        self.pops_size += new_pops.pops_size


    def get_best_acc(self, is_random=False):
        acc_list = []
        if is_random:
            for i in range(self.get_pop_size()):
                indi = self.get_individual(i)
                acc_list.append(indi.random_acc)
            return np.max(acc_list)
        
        for i in range(self.get_pop_size()):
            indi = self.get_individual(i)
            acc_list.append(indi.mean_acc)
        return np.max(acc_list)
    

    def get_sorted_index_order_by_acc(self, is_random=False):
        acc_list = []
        if is_random:
            for i in range(self.get_pop_size()):
                indi = self.get_individual(i)
                acc_list.append(indi.random_acc)
            arg_index = np.argsort(-1 * np.array(acc_list))
            return arg_index
        
        for i in range(self.get_pop_size()):
            indi = self.get_individual(i)
            acc_list.append(indi.mean_acc)
            arg_index = np.argsort(-1 * np.array(acc_list))
        return arg_index
    
    
    def __str__(self):
        str_ = []
        arg_index = self.get_sorted_index_order_by_acc()
        for i in arg_index:
            str_.append(str(self.get_individual(i)))
        return '\n'.join(str_)


class PopulationY:
    def __init__(self, pops_size, m_num_matrix=1, m_num_op_list=1):
        self.pops_size = pops_size
        self.pops = []
        for i in range(pops_size):
            indi = IndividualY(m_num_matrix, m_num_op_list)
            indi.initialize()
            self.pops.append(indi)

    
    def copy_info_from_Population(self, copy_population):
        assert len(self.pops) == len(copy_population.pops)
        for i, indi in enumerate(copy_population.pops):
            self.pops[i].indi = indi.indi
            self.pops[i].mean_acc = indi.mean_acc
    
    def get_individual(self, i):
        return self.pops[i]
    
    
    def get_pop_size(self):
        return len(self.pops)
    

    def set_populations(self, new_pops):
        self.pops = new_pops
        self.pops_size = len(new_pops)


    def merge_populations(self, new_pops):
        for indi in new_pops:
            self.pops.append(indi)
        self.pops_size += new_pops.pops_size

    def get_best_acc(self, dataset):
        acc_list = []
        for i in range(self.pops_size):
            indi = self.pops[i]
            acc_list.append(indi.mean_acc[dataset])
        return np.max(acc_list)
    

    def get_sorted_index_order_by_acc(self, dataset):
        acc_list = []
        for i in range(self.get_pop_size()):
            indi = self.pops[i]
            acc_list.append(indi.mean_acc[dataset])
        arg_index = np.argsort(-1 * np.array(acc_list))
        return arg_index
    
    
    def __str__(self):
        str_ = []
        for i in range(self.pops_size):
            str_.append(str(self.pops[i]))
        return '\n'.join(str_)




if __name__ == '__main__':
    # pop = PopulationX(pops_size=20, m_num_matrix=2, m_num_op_list=2)
    # print(pop)
    # print(str(pop))

    pop = PopulationY(pops_size=2, m_num_matrix=1, m_num_op_list=1)
    print(pop)

