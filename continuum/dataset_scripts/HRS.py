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
        self.data = [deque(np.nan_to_num(np.array(scipy.io.loadmat(
            os.path.join('data', 'HRS', 'normal_V2.mat'))['NORMAL'])))]
        self.f_samples = {0: (len(self.data[0]) - self.params.N) // 6}
        fault = scipy.io.loadmat(os.path.join('data', 'HRS', 'fault_V2.mat'))
        for i in range(1, 6):
            self.data.append(
                deque(np.nan_to_num(np.array(fault[f'Fault{i}']))))
            self.f_samples[i] = len(self.data[i]) // (6-i)

    def setup(self):
        self.test_set = []
        if self.scenario == 'nc':
            for cur in range(len(self.data)):
                test_label = np.zeros(self.f_samples[0], dtype=int)
                test_data = np.array([self.data[0].popleft()
                                     for _ in range(self.f_samples[0])])

                if cur != 0:
                    for i in range(cur):
                        segment = [self.data[i+1].popleft()
                                   for _ in range(self.f_samples[i+1])]
                        test_data = np.concatenate((test_data, segment))
                        test_label = np.concatenate(
                            (test_label, np.full(len(segment), i+1, dtype=int)))
                self.test_set.append((test_data, test_label))

        if self.scenario == 'vc':
            labels = [np.zeros(len(self.data[0]), dtype=int)]
            for i in range(1, len(self.data)):
                labels.append(np.full(len(self.data[i]), i, dtype=int))

            labels = np.concatenate(labels, axis=0)
            data = np.concatenate(self.data, axis=0)

            self.test_set = construct_ns_multiple_wrapper(
                data[:-1, :], labels[:-1], self.task_nums, 120, self.params.ns_type, self.params.ns_factor, plot=self.params.plot_sample)

    def s_sample(self, x_train, y_train):
        n_idx = np.where(y_train == 0)[0]
        f_idx = np.where(y_train != 0)[0]

        n_cls = len(n_idx)
        f_cls = len(f_idx)

        if self.params.n_r == 0 or n_cls == 0:
            target_f_size = f_cls
            sampled_f_idx = resample(
                f_idx, replace=False, n_samples=target_f_size, random_state=42)
            sampled_indices = sampled_f_idx
        elif self.params.f_r == 0 or f_cls == 0:
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
        x_train = np.array([self.data[0].popleft()
                           for _ in range(self.params.N)])
        y_train = np.zeros(len(x_train), dtype=int)

        return x_train, y_train

    def new_run(self, **kwargs):
        self.setup()
        return self.test_set

    def test_plot(self):
        test_ns(self.train_data[:6], self.train_label[:6], self.params.ns_type,
                self.params.ns_factor)
