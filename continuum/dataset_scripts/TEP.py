import os
import numpy as np
import random
from continuum.dataset_scripts.dataset_base import DatasetBase
import numpy as np
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns
from glob import iglob


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
        self.data = [np.concatenate([temp[0], normal], axis=0)]

        self.data += [temp[i][160:] for i in range(1, len(temp))]

    def setup(self):
        self.test_set = []
        if self.scenario == 'nc':
            for cur, t_set in enumerate(self.data):
                test_label = np.zeros(self.params.n, dtype=int)
                test_data = self.data[0][np.random.choice(
                    self.data[0].shape[0], size=self.params.n, replace=False)]

                if cur != 0:
                    n = self.params.f // cur
                    for i in range(cur):
                        start = self.params.n - self.params.f + n * i
                        end = start + n if i + 1 != cur else self.params.n

                        test_data[start:end] = t_set[np.random.choice(
                            t_set.shape[0], size=end - start, replace=False)]
                        test_label[start:end] = i + 1
                self.test_set.append((test_data, test_label))

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

        selected_indices = random.sample(range(len(y_train)), k=int(
            self.params.n * random.uniform(0.25, 0.50)))

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
