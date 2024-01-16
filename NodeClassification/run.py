import os
import datetime
import time
import subprocess
import utils

# model_list = ['gcn', 'graphsage', 'gin']
# dataset_list = ['cora', 'citeseer', 'chameleon', 'wikics']
# dataset_list = ['cora', 'citeseer', 'chameleon']
# graph_prune_ratio = [0.01, 0.1, 0.3, 0.5, 0.8, 0.9, 0.99]

# model_list = ['gcn', 'graphsage']
# dataset_list = ['pubmed']
# graph_prune_ratio = [0.8]

model_list = ['gcn', 'graphsage']
dataset_list = ['cora', 'citeseer']
graph_prune_ratio = [0.8]

# graph_prune_ratio = [0.01, 0.1, 0.3, 0.5, 0.8, 0.9, 0.99]
# dataset_list = ['actor', 'squirrel']
# dataset_list = ['actor']

ISOTIMEFORMAT = '%m%d_%H%M'
theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
theTime = str(theTime)

Task_time_start = time.perf_counter()

for model in model_list:
    for dataset in dataset_list:
        for g_prune in graph_prune_ratio:
            if model == 'gcn' or model == 'graphsage':
                os.system('/usr/bin/python3.8 main_pruning_imp.py' + ' --gpu=0' + ' --net=' +
                          model + ' --dataset=' + dataset + ' --graph-prune-ratio=' + str(g_prune))
            elif model == 'gin':
                os.system('/usr/bin/python3.8 main_gingat_imp.py' + ' --gpu=0' + ' --net=' + model +
                          ' --dataset=' + dataset + ' --graph-prune-ratio=' + str(g_prune))

print('\n>> All Tasks finish, total execution time: {}'.format(
    utils.time_format(time.perf_counter() - Task_time_start)))
