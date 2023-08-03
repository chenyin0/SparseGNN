import os
import datetime
import time
import subprocess

model_list = ['gcn', 'graphsage', 'gin']
# dataset_list = ['cora', 'citeseer', 'chameleon', 'wikics']
dataset_list = ['squirrel']

ISOTIMEFORMAT = '%m%d_%H%M'
theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
theTime = str(theTime)

Task_time_start = time.perf_counter()

for dataset in dataset_list:
    for model in model_list:
        if model == 'gcn' or model == 'graphsage':
            os.system('/usr/bin/python3.8 main_pruning_imp.py' + ' --gpu=1' + ' --net=' + model +
                      ' --dataset=' + dataset)
        elif model == 'gin':
            os.system('/usr/bin/python3.8 main_gingat_imp.py' + ' --gpu=1' + ' --net=' + model +
                      ' --dataset=' + dataset)

print('\n>> All Tasks finish, total execution time: {:.4}s'.format(time.perf_counter() -
                                                                   Task_time_start))
