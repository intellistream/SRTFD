import numpy as np
from continuum.dataset_scripts.dataset_base import DatasetBase
import scipy.io
import numpy as np
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns
from collections import deque

import os
import random
import csv


class CARLS_M(DatasetBase):
    def __init__(self, scenario, params):
        dataset = 'CARLS_M'
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
        }

        super(CARLS_M, self).__init__(dataset, scenario,
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
        file_paths = ['data/CARLS(multi-sensor)/CarlaTown01-30Vehicles-ML-fault-MS-2.csv',
                      'data/CARLS(multi-sensor)/CarlaTown02-30Vehicles-ML-fault-MS-2.csv',
                      'data/CARLS(multi-sensor)/CarlaTown03-30Vehicles-ML-fault-MS-2.csv']

        self.nc_data = []
        self.vc_data = []

        for path in file_paths:
            d, l = self.load_file(path)

            self.nc_data.append([d[l == label] if label == 0 else deque(
                d[l == label]) for label in np.unique(l)])

            self.vc_data.append((d, l))

    def setup(self, **kwargs):
        self.test_set = []
        self.cur_run = kwargs.get('run')

        if self.scenario == 'nc':
            data = self.nc_data[self.cur_run]
            for cur in range(len(data)):
                test_label = np.zeros(self.params.n, dtype=int)
                test_data = data[0][np.random.choice(
                    data[0].shape[0], size=self.params.n, replace=False)]

                if cur != 0:
                    n = self.params.f // cur
                    for i in range(cur):
                        start = self.params.n - self.params.f + n * i
                        end = start + n if i + 1 != cur else self.params.n
                        test_data[start:end] = [data[i+1].popleft()
                                                for _ in range(end-start)]
                        test_label[start:end] = i + 1
                self.test_set.append((test_data, test_label))
        elif self.scenario == 'vc':
            for i in range(self.task_nums):
                x, y = self.vc_data[random.randint(0, len(self.vc_data)-1)]

                # selected_indices = random.sample(range(len(y)), k=10000)

                # x = x[selected_indices]
                # y = y[selected_indices]

                x = x.reshape(self.task_nums, -1, 10)
                y = y.reshape(self.task_nums, -1)
                self.test_set.append((x[i], y[i]))

    def new_task(self, cur_task, **kwargs):
        x_train, y_train = self.test_set[cur_task]
        if self.scenario == 'nc' and cur_task != 0:
            nonzero_positions = np.nonzero(y_train)[0]
            x_train = x_train[nonzero_positions]
            y_train = y_train[nonzero_positions]
            # print(y_train)

        selected_indices = random.sample(range(len(y_train)), k=int(
            len(y_train) * random.uniform(0.5, 0.7)))

        x_train = x_train[selected_indices]
        y_train = y_train[selected_indices]

        labels = np.unique(y_train)

        return x_train, y_train, labels

    def init_kw(self):
        if self.scenario == 'nc':
            data = self.nc_data[self.cur_run]
            x_train = data[0][np.random.choice(
                data[0].shape[0], size=int(data[0].shape[0] * 0.01), replace=False)]
            y_train = np.zeros(len(x_train), dtype=int)
        elif self.scenario == 'vc':
            x, y = self.vc_data[self.cur_run]
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
