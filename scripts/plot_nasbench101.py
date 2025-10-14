'''
Descripttion: 
version: 1.0
Author: ZXL
Date: 2023-12-10 16:43:08
LastEditors: ZXL
LastEditTime: 2024-03-11 14:41:08
'''
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from get_data_from_101 import calculate_mean_acc

current_path = os.path.dirname(__file__)
hash_list_path = os.path.join(current_path + '/pkl','hash_list.pkl')
with open(hash_list_path, 'rb') as file:
    hash_list = pickle.load(file)
# computed_statisticst_path = os.path.join(current_path + '/pkl','computed_statistics.pkl')
# with open(computed_statisticst_path, 'rb') as file:
#     computed_statisticst = pickle.load(file)
computed_statisticst_top_path = os.path.join(current_path + '/pkl', 'computed_statisticst_top1000.pkl')
with open(computed_statisticst_top_path, 'rb') as file:
    computed_statisticst_top1000 = pickle.load(file)


# the figure of mean_acc of top 1000
# mean_acc_topk = []
# for i in range(1000):
#     mean_acc = calculate_mean_acc(computed_statisticst_top1000, hash_list[-(i+1)])
#     mean_acc_topk.append(mean_acc)

# plt.plot(mean_acc_topk)
# plt.xlabel('the top 1000 architectures')
# plt.ylabel('The average accuracy')
# plt.grid()
# plt.savefig(current_path + '/figures/PDF-version/mean_acc_top1000.pdf')
# plt.savefig(current_path + '/figures/PNG-version/mean_acc_top1000.png')
# plt.show()

mean_acc_topk = []
for i in range(1000):
    mean_acc = calculate_mean_acc(computed_statisticst_top1000, hash_list[-(i+1)])
    mean_acc_topk.append(mean_acc)

max = 0
for i in range(1000):
    if mean_acc_topk[i] < 0.9431:
        if mean_acc_topk[i] > max:
            max = mean_acc_topk[i]

print(max)
# print(np.max(mean_acc_list))
# print(np.min(mean_acc_list))
