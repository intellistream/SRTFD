import os
import numpy as np
import random
from continuum.dataset_scripts.dataset_base import DatasetBase
import numpy as np
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns
from glob import iglob
from collections import deque


class TEP(DatasetBase):
    def __init__(self, scenario, params):
        dataset = 'TEP'
        if scenario == 'ni':
            num_tasks = len(params.ns_factor)
        else:
            num_tasks = params.num_tasks
        super(TEP, self).__init__(dataset, scenario,
                                  num_tasks, params.num_runs, params)

    def load_file(self, filename):
        rows = []
        with open(filename) as f:
            for line in f:
                line = [float(s) for s in line.split()]
                rows.append(np.array(line, dtype=np.double))
        return np.vstack(rows)

    def download_load(self):
        file_paths = sorted(iglob(os.path.join('data', 'TEP', '*.dat')))

        temp = [self.load_file(file) for file in file_paths]

        normal = np.concatenate([temp[i][:160]
                                for i in range(1, len(temp))], axis=0)
        self.data = [deque(np.concatenate([temp[0], normal], axis=0))]

        self.f_samples = {0: (len(self.data[0]) - self.params.N) // 22}

        for i in range(1, 22):
            self.data.append(deque(temp[i][160:]))
            self.f_samples[i] = len(self.data[i]) // (22-i)

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
                data, labels, self.task_nums, 52, self.params.ns_type, self.params.ns_factor, plot=self.params.plot_sample)

    def new_task(self, cur_task, **kwargs):
        x_train, y_train = self.test_set[cur_task]

        if cur_task != 0:
            nonzero_positions = np.nonzero(y_train)[0]
            x_train = x_train[nonzero_positions]
            y_train = y_train[nonzero_positions]

        selected_indices = random.sample(range(len(y_train)), k=int(
            len(y_train) * random.uniform(0.5, 0.7)))

        x_train = x_train[selected_indices]
        y_train = y_train[selected_indices]

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
