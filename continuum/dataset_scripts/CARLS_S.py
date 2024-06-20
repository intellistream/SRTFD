import numpy as np
from continuum.dataset_scripts.dataset_base import DatasetBase
import scipy.io
import numpy as np
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns

import os
import random
import csv


class CARLS_S(DatasetBase):
    def __init__(self, scenario, params):
        dataset = 'CARLS_S'
        if scenario == 'ni':
            num_tasks = len(params.ns_factor)
        else:
            num_tasks = params.num_tasks

        self.label_mapping = {
            'Normal': 0,
            'Fault-1': 1,
            'Fault-2': 2,
            'Fault-3': 3,
            'Fault-4': 4,
            'Fault-5': 5,
            'Fault-6': 6,
            'Fault-7': 7,
            'Fault-8': 8,
            'Fault-9': 9
        }

        super(CARLS_S, self).__init__(dataset, scenario,
                                      num_tasks, params.num_runs, params)

    def load_file(self, filename):
        data = []
        labels = []
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # header

            for row in reader:
                data.append([float(val) for val in row[:-1]])
                labels.append(self.label_mapping[row[-1].strip()])

        return np.array(data), np.array(labels)

    def download_load(self):
        file_paths = ['data/CARLS(single-sensor)/CarlaTown01-30Vehicles-ML-fault-SS-2.csv',
                      'data/CARLS(single-sensor)/CarlaTown02-30Vehicles-ML-fault-SS-2.csv',
                      'data/CARLS(single-sensor)/CarlaTown03-30Vehicles-ML-fault-SS-2.csv']

        self.data = [self.load_file(file) for file in file_paths]

    def setup(self, **kwargs):
        self.test_set = []
        self.cur_run = kwargs.get('run')
        if self.scenario == 'nc':
            x, y = self.data[self.cur_run]
            normal = x[y == 0]
            for cur in range(self.params.num_tasks):
                test_label = np.zeros(self.params.n, dtype=int)
                test_data = normal[np.random.choice(
                    normal.shape[0], size=self.params.n, replace=False)]

                if cur != 0:
                    n = self.params.f // cur
                    for j in range(cur):
                        start = self.params.n - self.params.f + n * j
                        end = start + n if j + 1 != cur else self.params.n

                        fault = x[y == (j+1)]

                        test_data[start:end] = fault[np.random.choice(
                            fault.shape[0], size=end - start, replace=False)]
                        test_label[start:end] = j + 1
                self.test_set.append((test_data, test_label))
        elif self.scenario == 'vc':
            for i in range(self.task_nums):
                x, y = self.data[random.randint(0, len(self.data)-1)]
                x = x.reshape(self.task_nums, -1, 10)
                y = y.reshape(self.task_nums, -1)
                self.test_set.append((x[i], y[i]))

    def new_task(self, cur_task, **kwargs):
        x_train, y_train = self.test_set[cur_task]

        selected_indices = random.sample(range(len(y_train)), k=int(
            self.params.n * random.uniform(0.25, 0.50)))

        x_train = x_train[selected_indices]
        y_train = y_train[selected_indices]

        labels = np.unique(y_train)

        return x_train, y_train, labels

    def init_kw(self):
        x, y = self.data[self.cur_run]
        normal_idx = y == 0
        x_train = x[normal_idx][np.random.choice(
            x[normal_idx].shape[0], size=int(x[normal_idx].shape[0] * 0.01), replace=False)]
        y_train = np.zeros(len(x_train), dtype=int)

        return x_train, y_train

    def new_run(self, **kwargs):
        self.setup(run=kwargs.get('cur_run'))
        return self.test_set

    def test_plot(self):
        test_ns(self.train_data[:6], self.train_label[:6], self.params.ns_type,
                self.params.ns_factor)
