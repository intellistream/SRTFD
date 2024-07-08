import numpy as np
from continuum.dataset_scripts.dataset_base import DatasetBase
import scipy.io
import numpy as np
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns

import os
import random

class HRS(DatasetBase):
    def __init__(self, scenario, params):
        dataset = 'HRS'
        if scenario == 'ni':
            num_tasks = len(params.ns_factor)
        else:
            num_tasks = params.num_tasks
        super(HRS, self).__init__(dataset, scenario,
                                  num_tasks, params.num_runs, params)

    def download_load(self):
        self.data = [np.nan_to_num(np.array(scipy.io.loadmat(
            os.path.join('data', 'HRS', 'normal_V2.mat'))['NORMAL']))]
        fault = scipy.io.loadmat(os.path.join('data', 'HRS', 'fault_V2.mat'))
        for i in range(1, 6):
            self.data.append(np.nan_to_num(np.array(fault[f'Fault{i}'])))
            
    def setup(self):
        self.test_set = []
        if self.scenario == 'nc':
            num = 5
            for cur, t_set in enumerate(self.data):
                test_label = np.zeros(self.params.n, dtype=int)
                test_data = self.data[0][np.random.choice(
                    self.data[0].shape[0], size=self.params.n, replace=False)]
                
                if cur != 0:
                    n = len(t_set) // num 
                    for i in range(cur):
                        test_data[n*i:n*(i+1)] = t_set[n*i:n*(i+1)]
                        test_label[n*i:n*(i+1)] = i + 1
                self.test_set.append((test_data, test_label))
            num = - 1

        if self.scenario == 'vc':

            labels = [np.zeros(self.data[0].shape[0], dtype=int)]
            for i in range(1, len(self.data)):
                labels.append(np.full(self.data[i].shape[0], i, dtype=int))

            labels = np.concatenate(labels, axis=0)
            data = np.concatenate(self.data, axis=0)

            self.test_set = construct_ns_multiple_wrapper(
                data, labels, self.task_nums, 52, self.params.ns_type, self.params.ns_factor, plot=self.params.plot_sample)


    def new_task(self, cur_task, **kwargs):
        x_train, y_train = self.test_set[cur_task]
        
        if cur_task !=0:
            nonzero_positions = np.nonzero(y_train)[0]
            x_train = x_train[nonzero_positions]
            y_train = y_train[nonzero_positions]
            #print(y_train)

        selected_indices = random.sample(range(len(y_train)), k=int(
            len(y_train) * random.uniform(0.5, 0.7)))

        x_train = x_train[selected_indices]
        y_train = y_train[selected_indices]

        labels = np.unique(y_train)

        return x_train, y_train, labels

    def init_kw(self):
        x_train = self.data[0][np.random.choice(
            self.data[0].shape[0], size=int(self.data[0].shape[0] * 0.01), replace=False)]
        y_train = np.zeros(len(x_train), dtype=int)

        return x_train, y_train

    def new_run(self, **kwargs):
        self.setup()
        return self.test_set

    def test_plot(self):
        test_ns(self.train_data[:6], self.train_label[:6], self.params.ns_type,
                self.params.ns_factor)
