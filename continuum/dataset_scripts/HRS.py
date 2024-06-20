import glob
from PIL import Image
import numpy as np
from continuum.dataset_scripts.dataset_base import DatasetBase
import time
from continuum.data_utils import shuffle_data
import scipy.io
import numpy as np
from torchvision import datasets
from continuum.data_utils import create_task_composition, load_task_with_labels, shuffle_data
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns

# data1 = scipy.io.loadmat('D:\online-continual-learning-main\online-continual-learning-main\datasets\\fault.mat')

# data2 = scipy.io.loadmat('D:\online-continual-learning-main\online-continual-learning-main\datasets\\normal.mat')

# # 打印字典中的键
# y_train = []
# y_test = []
# x_train = []
# x_test = []
    
# Fault1 = np.nan_to_num(data1['Fault1'])
# num_samples1 = Fault1.shape[0]
# permuted_indices1 = np.random.permutation(num_samples1)
# split_point1 = int(num_samples1 * 0.7)
# x_train1 = Fault1[permuted_indices1[:split_point1]]
# y_train.extend([1] * x_train1.shape[0])
# x_train.extend(x_train1)
# x_test1 = Fault1[permuted_indices1[split_point1:]]
# y_test.extend([1] * x_test1.shape[0])
# x_test.extend(x_test1)


# Fault2 = np.nan_to_num(data1['Fault2'])
# num_samples2 = Fault2.shape[0]
# permuted_indices2 = np.random.permutation(num_samples2)
# split_point2 = int(num_samples2 * 0.7)
# x_train2 = Fault2[permuted_indices2[:split_point2]]
# y_train.extend([2] * x_train2.shape[0])
# x_test2 = Fault2[permuted_indices2[split_point2:]]
# y_test.extend([2] * x_test2.shape[0])
# x_train.extend(x_train2)
# x_test.extend(x_test2)


# Fault3 = np.nan_to_num(data1['Fault3'])
# num_samples3 = Fault3.shape[0]
# permuted_indices3 = np.random.permutation(num_samples3)
# split_point3 = int(num_samples3 * 0.7)
# x_train3 = Fault3[permuted_indices3[:split_point3]]
# y_train.extend([3] * x_train3.shape[0])
# x_test3 = Fault3[permuted_indices3[split_point3:]]
# y_test.extend([3] * x_test3.shape[0])
# x_train.extend(x_train3)
# x_test.extend(x_test3)

# Fault4 = np.nan_to_num(data1['Fault4'])
# num_samples4 = Fault4.shape[0]
# permuted_indices4 = np.random.permutation(num_samples4)
# split_point4 = int(num_samples4 * 0.7)
# x_train4 = Fault4[permuted_indices4[:split_point4]]
# y_train.extend([4] * x_train4.shape[0])
# x_test4 = Fault4[permuted_indices4[split_point4:]]
# y_test.extend([4] * x_test4.shape[0])
# x_train.extend(x_train4)
# x_test.extend(x_test4)

# Fault5 = np.nan_to_num(data1['Fault5'])
# num_samples5 = Fault5.shape[0]
# permuted_indices5 = np.random.permutation(num_samples5)
# split_point5 = int(num_samples5 * 0.7)
# x_train5 = Fault5[permuted_indices5[:split_point5]]
# y_train.extend([5] * x_train5.shape[0])
# x_test5 = Fault5[permuted_indices5[split_point5:]]
# y_test.extend([5] * x_test5.shape[0])
# x_train.extend(x_train5)
# x_test.extend(x_test5)

# Normal = np.nan_to_num(data2['NORMAL'])
# num_samples6 = Normal.shape[0]
# permuted_indices6 = np.random.permutation(num_samples6)
# split_point6 = int(num_samples6 * 0.7)
# x_train6 = Normal[permuted_indices6[:split_point6]]
# y_train.extend([0] * x_train6.shape[0])
# x_test6 = Normal[permuted_indices6[split_point6:]]
# y_test.extend([0] * x_test6.shape[0])
# x_train.extend(x_train6)
# x_test.extend(x_test6)

# data_to_save = {
#     'x_train': x_train,
#     'x_test': x_test,
#     'y_train': y_train,
#     'y_test': y_test,
# }

# scipy.io.savemat('HRSs.mat', data_to_save)



# class HRS(DatasetBase):
#     """
#     tasks_nums is predefined and it depends on the ns_type.
#     """
#     def __init__(self, scenario, params):  # scenario refers to "ni" or "nc"
#         dataset = 'openloris'
#         self.ns_type = params.ns_type
#         task_nums = params.num_tasks
#         super(HRS, self).__init__(dataset, scenario, task_nums, params.num_runs, params)


#     def download_load(self):
#         s = time.time()
#         import os
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         file_path = os.path.join(script_dir, 'HRSs.mat')
#         data = scipy.io.loadmat(file_path)
#        # data = scipy.io.loadmat(r'D:\online-continual-learning-main\online-continual-learning-main\datasets\HRSs.mat')

#         self.train_set = []
#         self.train_set.append((np.array(data['x_train']), np.array(data['y_train'])))
#         self.test_set.append((np.array(data['x_test']), np.array(data['y_test'])))
#         e = time.time()
#       #  print('loading time: {}'.format(str(e - s)))

#     def new_run(self, **kwargs):
#         pass

#     def new_task(self, cur_task, **kwargs):
#         train_x, train_y = self.train_set[cur_task]
#         # get val set
#         train_x_rdm, train_y_rdm = shuffle_data(train_x, train_y)
#         val_size = int(len(train_x_rdm) * self.params.val_size)
#         val_data_rdm, val_label_rdm = train_x_rdm[:val_size], train_y_rdm[:val_size]
#         train_data_rdm, train_label_rdm = train_x_rdm[val_size:], train_y_rdm[val_size:]
#         self.val_set.append((val_data_rdm, val_label_rdm))
#         labels = self.train_set[cur_task]
#        # labels = set(train_label_rdm)
#         return train_data_rdm, train_label_rdm, labels

#     def setup(self, **kwargs):
#         pass






class HRS(DatasetBase):
    def __init__(self, scenario, params):
        dataset = 'HRS'
        if scenario == 'ni':
            num_tasks = len(params.ns_factor)
        else:
            num_tasks = params.num_tasks
        super(HRS, self).__init__(dataset, scenario, num_tasks, params.num_runs, params)


    def download_load(self):
        # dataset_train = datasets.CIFAR10(root=self.root, train=True, download=True)
        # self.train_data = dataset_train.data
        # self.train_label = np.array(dataset_train.targets)
        # dataset_test = datasets.CIFAR10(root=self.root, train=False, download=True)
        # self.test_data = dataset_test.data
        # self.test_label = np.array(dataset_test.targets)
       
        
        s = time.time()
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path1 = os.path.join(script_dir, 'data_test.mat')
        data_test = scipy.io.loadmat(file_path1)
        
        file_path2 = os.path.join(script_dir, 'task_data.mat')
        data_train = scipy.io.loadmat(file_path2)
        file_path3 = os.path.join(script_dir, 'task_label.mat')
        label_train = scipy.io.loadmat(file_path3)
        
        self.test_data = data_test['data_test']
        self.test_label = data_test['labale_test']
        self.test_label = self.test_label.reshape(-1)
        
        
        self.train_data = data_train
        self.train_label = label_train
        # train_data1 = np.array(data['x_train'])
        # train_label1 = np.array(data['y_train'])
        # train_label1 = train_label1.reshape(-1)
        # test_data1 = np.array(data['x_test'])
        # test_label1 = np.array(data['y_test'])
        # test_label1 = test_label1.reshape(-1)
        e = time.time()
              # print('loading time: {}'.format(str(e - s)))

    def setup(self):
        if self.scenario == 'ni':
            self.train_set, self.val_set, self.test_set = construct_ns_multiple_wrapper(self.train_data,
                                                                                        self.train_label,
                                                                                        self.test_data, self.test_label,
                                                                                        self.task_nums, 6,
                                                                                        self.params.val_size,
                                                                                        self.params.ns_type, self.params.ns_factor,
                                                                                        plot=self.params.plot_sample)
        elif self.scenario == 'nc':
            
            x_test = []
            y_test = []
            cur = 1
            for j in range(6):
                for i in range(cur):
                    index = np.where(self.test_label == i)
                    x_test.append(self.test_data[index])
                    y_test.append(self.test_label[index])
                cur = cur  + 1
                x_test1 = np.concatenate(x_test, axis=0)
                y_test1 = np.concatenate(y_test, axis=0)
                print(len(y_test1))
                self.test_set.append((x_test1,  y_test1))
            
           # for i in range(6):
                #index = np.where(self.test_label == i)
               # x_test, y_test =self.test_data[index] , self.test_label[index]
            # self.test_set.append((self.test_data, self.test_label))
            # self.task_labels = create_task_composition(class_nums=6, num_tasks=self.task_nums, fixed_order=self.params.fix_order)
            # self.test_set = []
            # for labels in self.task_labels:
            #     x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)
            #     self.test_set.append((x_test, y_test))
        else:
            raise Exception('wrong scenario')

    def new_task(self, cur_task, **kwargs):
        if self.scenario == 'ni':
            x_train, y_train = self.train_set[cur_task]
            
            labels = set(y_train)
        elif self.scenario == 'nc':
            if cur_task == 0:
                key = 'task1'
            elif cur_task == 1:
                key = 'task2'
            elif cur_task == 2:
                key = 'task3'
            elif cur_task == 3:
                key = 'task4'
            elif cur_task == 4:
                key = 'task5'
            elif cur_task == 5:
                key = 'task6'
            labels = np.unique(self.train_label[key].reshape(-1))
            
            y_train = self.train_label[key].reshape(-1)
            x_train = self.train_data[key]
            
            #x_train, y_train = self.train_data,self.train_label
            
            #labels = self.task_labels[cur_task]
           # x_train, y_train = load_task_with_labels(self.train_data, self.train_label, labels)
        return x_train, y_train, labels

    def new_run(self, **kwargs):
        self.setup()
        return self.test_set

    def test_plot(self):
        test_ns(self.train_data[:6], self.train_label[:6], self.params.ns_type,
                                                          self.params.ns_factor)

