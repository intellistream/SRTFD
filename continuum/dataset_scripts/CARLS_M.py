import numpy as np
from continuum.dataset_scripts.dataset_base import DatasetBase
import scipy.io
import numpy as np
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns
from collections import deque
from sklearn.utils import resample

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
        self.f_samples = []

        for path in file_paths:
            d, l = self.load_file(path)

            self.nc_data.append([deque(d[l == label])
                                for label in np.unique(l)])

            self.f_samples.append(
                {0: (len(self.nc_data[-1][0]) - self.params.N) // 5})
            for i in range(1, len(self.nc_data[-1])):
                self.f_samples[-1][i] = len(self.nc_data[-1]
                                            [i]) // (len(self.nc_data[-1]) - i)
            self.vc_data.append((d, l))

    def setup(self, **kwargs):
        self.test_set = []
        self.cur_run = kwargs.get('run')

        if self.scenario == 'nc':
            data = self.nc_data[self.cur_run]
            for cur in range(len(data)):
                test_label = np.zeros(
                    self.f_samples[self.cur_run][0], dtype=int)
                test_data = np.array([data[0].popleft()
                                     for _ in range(self.f_samples[self.cur_run][0])])

                if cur != 0:
                    for i in range(cur):
                        segment = [data[i+1].popleft()
                                   for _ in range(self.f_samples[self.cur_run][i+1])]
                        test_data = np.concatenate((test_data, segment))
                        test_label = np.concatenate(
                            (test_label, np.full(len(segment), i+1, dtype=int)))

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

    def s_sample(self, x_train, y_train):
        n_idx = np.where(y_train == 0)[0]
        f_idx = np.where(y_train != 0)[0]

        n_cls = len(n_idx)
        f_cls = len(f_idx)

        if self.params.n_r == 0:
            target_f_size = f_cls
            sampled_f_idx = resample(
                f_idx, replace=False, n_samples=target_f_size, random_state=42)
            sampled_indices = sampled_f_idx
        elif self.params.f_r == 0:
            target_n_size = n_cls
            sampled_n_idx = resample(
                n_idx, replace=False, n_samples=target_n_size, random_state=42)
            sampled_indices = sampled_n_idx
        else:
            target_n_size = int(f_cls * self.params.n_r /
                                self.params.f_r) if self.params.f_r > 0 else n_cls
            target_f_size = int(n_cls * self.params.f_r /
                                self.params.n_r) if self.params.n_r > 0 else f_cls

            target_n_size = min(n_cls, target_n_size)
            target_f_size = min(f_cls, target_f_size)

            sampled_n_idx = resample(
                n_idx, replace=False, n_samples=target_n_size, random_state=42)
            sampled_f_idx = resample(
                f_idx, replace=False, n_samples=target_f_size, random_state=42)

            sampled_indices = np.concatenate([sampled_n_idx, sampled_f_idx])

        np.random.shuffle(sampled_indices)

        return x_train[sampled_indices], y_train[sampled_indices]

    def new_task(self, cur_task, **kwargs):
        x_train, y_train = self.test_set[cur_task]

        if self.scenario == 'nc':
            if cur_task != 0:
                nonzero_positions = np.nonzero(y_train)[0]
                x_train = x_train[nonzero_positions]
                y_train = y_train[nonzero_positions]
                # print(y_train)

        selected_indices = random.sample(range(len(y_train)), k=int(
            len(y_train) * random.uniform(0.5, 0.7)))

        x_train = x_train[selected_indices]
        y_train = y_train[selected_indices]

        if self.scenario == 'vc':
            x_train, y_train = self.s_sample(x_train, y_train)

        labels = np.unique(y_train)

        return x_train, y_train, labels

    def init_kw(self):
        if self.scenario == 'nc':
            data = self.nc_data[self.cur_run]
            x_train = np.array([data[0].popleft()
                                for _ in range(self.params.N)])
            y_train = np.zeros(len(x_train), dtype=int)
        elif self.scenario == 'vc':
            x, y = self.vc_data[self.cur_run]
            x_train = x[:self.params.N]
            y_train = y[:self.params.N]

            self.vc_data[self.cur_run] = (x[self.params.N:], y[self.params.N:])

        return x_train, y_train

    def new_run(self, **kwargs):
        self.setup(run=kwargs.get('cur_run'))
        return self.test_set

    def test_plot(self):
        test_ns(self.train_data[:6], self.train_label[:6], self.params.ns_type,
                self.params.ns_factor)
