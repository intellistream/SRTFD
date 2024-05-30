# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:48:10 2024

@author: Zhao Dandan
"""
import scipy.io
import math
from collections import defaultdict
import numpy as np
data1 = scipy.io.loadmat('D:\online-continual-learning-main\online-continual-learning-main\datasets\\fault.mat')
data25 = scipy.io.loadmat('D:\online-continual-learning-main\online-continual-learning-main\datasets\\fault25.mat')
data11 = scipy.io.loadmat('D:\online-continual-learning-main\online-continual-learning-main\datasets\\fault_duo.mat')
data2 = scipy.io.loadmat('D:\online-continual-learning-main\online-continual-learning-main\datasets\\normal.mat')

data = []
data.append(data1)
data.append(data25)
data.append(data11)

name = ['s','s','s','Fault1', 'Fault2', 'Fault3', 'Fault4', 'Fault5']
count = 1
dict1 = {}
for (key1, name1) in zip(data[0].keys(), name):
    if 3 <= count <= 6:
        #data[0][key1]
        print(key1, name1)
        
        F1 = np.array(data[0][key1])
        F1 = np.nan_to_num(F1)
        F2 = np.array(data[1][key1])
        F2 = np.nan_to_num(F2)
        F = np.concatenate((F1, F2))
        F3 = np.array(data[2][key1])
        F3 = np.nan_to_num(F3)
        F = np.concatenate((F, F3))
        dict1.update({key1:F})
    else:
        data[0][key1] = np.nan_to_num(data[0][key1])
        dict1.update({key1:data[0][key1]})
    count = count+1
    

data2['NORMAL'] = np.nan_to_num(data2['NORMAL'])
dict1.update({'normal':data2['NORMAL']})

train ={}
test = {}
testset=[]
count = 1
for key in dict1:
    if 3 <= count:
        if count == 7 or count == 8:
            data = dict1[key]
            num_samples = data.shape[0]
            permuted_indices = np.random.permutation(num_samples)
            split_point = int(num_samples * 0.8)
            x_train = data[permuted_indices[:split_point]]
            x_test = data[permuted_indices[split_point:]]
            train.update({key:x_train})
            test.update({key:x_test})
            testset.append(x_test)
        elif count == 4:
            data = dict1[key]
            num_samples = data.shape[0]
            permuted_indices = np.random.permutation(num_samples)
            split_point = int(num_samples * 0.9)
            x_train = data[permuted_indices[:split_point+2]]
            x_test = data[permuted_indices[split_point+2:]]
            train.update({key:x_train})
            test.update({key:x_test})
            testset.append(x_test)
        elif count==5:
            data = dict1[key]
            num_samples = data.shape[0]
            permuted_indices = np.random.permutation(num_samples)
            split_point = int(num_samples * 0.9)
            x_train = data[permuted_indices[:split_point-1]]
            x_test = data[permuted_indices[split_point-1:]]
            train.update({key:x_train})
            test.update({key:x_test})
            testset.append(x_test)
        else:
            data = dict1[key]
            num_samples = data.shape[0]
            permuted_indices = np.random.permutation(num_samples)
            split_point = int(num_samples * 0.9)
            x_train = data[permuted_indices[:split_point]]
            x_test = data[permuted_indices[split_point:]]
            train.update({key:x_train})
            test.update({key:x_test})
            testset.append(x_test)
    count = count+1
del testset[0]
del test['__globals__']
del train['__globals__']

last_key = list(test.keys())[-1]
last_value = test.pop(last_key)
test = {last_key: last_value, **test}

labale_test= []
last_key = list(train.keys())[-1]
last_value = train.pop(last_key)
train = {last_key: last_value, **train}


# data_test = np.concatenate((test['normal'],test['Fault1']))
# data_test = np.concatenate((data_test,test['Fault2']))
# data_test = np.concatenate((data_test,test['Fault3']))
# data_test = np.concatenate((data_test,test['Fault4']))
# data_test = np.concatenate((data_test,test['Fault5']))
# labale_test.extend([0] * len(test['normal']))
# labale_test.extend([1] * len(test['Fault1']))
# labale_test.extend([2] * len(test['Fault2']))
# labale_test.extend([3] * len(test['Fault3']))
# labale_test.extend([4] * len(test['Fault4']))
# labale_test.extend([5] * len(test['Fault5']))

# data_test = {
#     'data_test': data_test, 
#     'labale_test':labale_test}

# scipy.io.savemat('data_test.mat', data_test)
#scipy.io.savemat('labale_test.mat', labale_test)

x_nor = np.split(train['normal'], 6)
x_f1 = np.split(train['Fault1'], 5)
x_f2 = np.split(train['Fault2'], 4)
x_f3 = np.split(train['Fault3'], 3)
x_f4 = np.split(train['Fault4'], 2)
x_f5 = np.split(train['Fault4'], 1)
#data_train.append(train['normal'])

data_train_t1 = []
label_train_t1 = []
data_train_t2 = []
label_train_t2 = []
data_train_t3 = []
label_train_t3 = []
data_train_t4 = []
label_train_t4 = []
data_train_t5 = []
label_train_t5 = []
data_train_t6 = []
label_train_t6 = []

task_data = {}
task_label = {}


#nor
data_train_t1 = x_nor[0]
label_train_t1.extend([0] * data_train_t1.shape[0])
task_data.update({'task1':data_train_t1})
task_label.update({'task1':label_train_t1})

#nor+fault1
data_train_t2 =  np.concatenate((x_nor[1][0:len(x_nor[1])-len(x_f1[0])],x_f1[0])) 
label_train_t2.extend([0] * len(x_nor[1][0:len(x_nor[1])-len(x_f1[0])]))
label_train_t2.extend([1] * len(x_f1[0]))

task_data.update({'task2':data_train_t2})
task_label.update({'task2':label_train_t2})

#nor+fault1+fault2
data_train_t3 =  np.concatenate((x_nor[2][0:len(x_nor[2])-len(x_f1[1])-len(x_f2[0])],x_f1[1])) 
data_train_t3 =  np.concatenate((data_train_t3,x_f2[0])) 

label_train_t3.extend([0] * len(x_nor[2][0:len(x_nor[2])-len(x_f1[1])-len(x_f2[0])]))
label_train_t3.extend([1] * len(x_f1[1]))
label_train_t3.extend([2] * len(x_f2[0]))

task_data.update({'task3':data_train_t3})
task_label.update({'task3':label_train_t3})

#nor+fault1+fault2+fault3
data_train_t4 =  np.concatenate((x_nor[3][0:len(x_nor[3])-len(x_f1[2])-len(x_f2[1])-len(x_f3[0])],x_f1[2])) 
data_train_t4 =  np.concatenate((data_train_t4,x_f2[1])) 
data_train_t4 =  np.concatenate((data_train_t4,x_f3[0])) 

label_train_t4.extend([0] * len(x_nor[2][0:len(x_nor[3])-len(x_f1[2])-len(x_f2[1])-len(x_f3[0])]))
label_train_t4.extend([1] * len(x_f1[2]))
label_train_t4.extend([2] * len(x_f2[1]))
label_train_t4.extend([3] * len(x_f3[0]))
task_data.update({'task4':data_train_t4})
task_label.update({'task4':label_train_t4})



#nor+fault1+fault2+fault3+fault4
data_train_t5 =  np.concatenate((x_nor[4][0:len(x_nor[4])-len(x_f1[2])-len(x_f2[1])-len(x_f3[0])-len(x_f4[0])],x_f1[3])) 
data_train_t5 =  np.concatenate((data_train_t5,x_f2[2])) 
data_train_t5 =  np.concatenate((data_train_t5,x_f3[1])) 
data_train_t5 =  np.concatenate((data_train_t5,x_f4[0])) 

label_train_t5.extend([0] * len(x_nor[4][0:len(x_nor[4])-len(x_f1[3])-len(x_f2[2])-len(x_f3[1])-len(x_f4[0])]))
label_train_t5.extend([1] * len(x_f1[3]))
label_train_t5.extend([2] * len(x_f2[2]))
label_train_t5.extend([3] * len(x_f3[1]))
label_train_t5.extend([4] * len(x_f4[0]))
task_data.update({'task5':data_train_t5})
task_label.update({'task5':label_train_t5})



#nor+fault1+fault2+fault3+fault4+fault5
data_train_t6 =  np.concatenate((x_nor[5][0:len(x_nor[5])-len(x_f1[4])-len(x_f2[3])-len(x_f3[2])-len(x_f4[1])-len(x_f5[0])],x_f1[4])) 
data_train_t6 =  np.concatenate((data_train_t6,x_f2[3])) 
data_train_t6 =  np.concatenate((data_train_t6,x_f3[2])) 
data_train_t6 =  np.concatenate((data_train_t6,x_f4[1])) 
data_train_t6 =  np.concatenate((data_train_t6,x_f5[0])) 

label_train_t6.extend([0] * len(x_nor[5][0:len(x_nor[5])-len(x_f1[4])-len(x_f2[3])-len(x_f3[2])-len(x_f4[1])-len(x_f5[0])]))
label_train_t6.extend([1] * len(x_f1[4]))
label_train_t6.extend([2] * len(x_f2[3]))
label_train_t6.extend([3] * len(x_f3[2]))
label_train_t6.extend([4] * len(x_f4[1]))
label_train_t6.extend([5] * len(x_f5[0]))
task_data.update({'task6':data_train_t6})
task_label.update({'task6':label_train_t6})


scipy.io.savemat('task_data.mat', task_data)
scipy.io.savemat('task_label.mat', task_label)

# # 打印字典中的键
# y_train = []
# y_test = []
# x_train = []
# x_test = []
    
# ##2024年label fault
# Fault1 = np.nan_to_num(data1['Fault1'])
# num_samples1 = Fault1.shape[0]
# permuted_indices1 = np.random.permutation(num_samples1)
# split_point1 = int(num_samples1 * 0.9)
# x_train1 = Fault1[permuted_indices1[:split_point1]]
# y_train.extend([1] * x_train1.shape[0])
# x_train.extend(x_train1)
# x_test1 = Fault1[permuted_indices1[split_point1:]]
# y_test.extend([1] * x_test1.shape[0])
# x_test.extend(x_test1)


# #2025年的label fault
# Fault251 = np.nan_to_num(data25['Fault1'])
# num_samples251 = Fault251.shape[0]
# permuted_indices251 = np.random.permutation(num_samples251)
# split_point251 = int(num_samples251 * 0.9)
# x_train1 = Fault251[permuted_indices251[:split_point251]]
# y_train.extend([1] * x_train1.shape[0])
# x_train.extend(x_train1)
# x_test1 = Fault1[permuted_indices251[split_point251:]]
# y_test.extend([1] * x_test1.shape[0])
# x_test.extend(x_test1)
# print(len(x_train1))


# #2025年的扩充的fault
# Fault11 = np.nan_to_num(data11['Fault1'])
# num_samples11 = Fault11.shape[0]
# permuted_indices11 = np.random.permutation(num_samples11)
# split_point11 = int(num_samples11 * 0.9)
# x_train1 = Fault11[permuted_indices11[:split_point11]]
# y_train.extend([1] * x_train1.shape[0])
# x_train.extend(x_train1)
# x_test1 = Fault11[permuted_indices11[split_point11:]]
# y_test.extend([1] * x_test1.shape[0])
# x_test.extend(x_test1)

# print(len(x_train1))


# Fault2 = np.nan_to_num(data1['Fault2'])
# num_samples2 = Fault2.shape[0]
# permuted_indices2 = np.random.permutation(num_samples2)
# split_point2 = int(num_samples2 * 0.9)
# x_train2 = Fault2[permuted_indices2[:split_point2]]
# y_train.extend([2] * x_train2.shape[0])
# x_test2 = Fault2[permuted_indices2[split_point2:]]
# y_test.extend([2] * x_test2.shape[0])
# x_train.extend(x_train2)
# x_test.extend(x_test2)


# Fault252 = np.nan_to_num(data25['Fault2'])
# num_samples252 = Fault252.shape[0]
# permuted_indices252 = np.random.permutation(num_samples252)
# split_point252 = int(num_samples252 * 0.9)
# x_train2 = Fault252[permuted_indices252[:split_point252]]
# y_train.extend([2] * x_train2.shape[0])
# x_test2 = Fault252[permuted_indices252[split_point252:]]
# y_test.extend([2] * x_test2.shape[0])
# x_train.extend(x_train2)
# x_test.extend(x_test2)



# Fault21 = np.nan_to_num(data11['Fault2'])
# num_samples21 = Fault21.shape[0]
# permuted_indices21 = np.random.permutation(num_samples21)
# split_point21 = int(num_samples21 * 0.9)
# x_train2 = Fault21[permuted_indices21[:split_point21]]
# y_train.extend([2] * x_train2.shape[0])
# x_test2 = Fault21[permuted_indices21[split_point21:]]
# y_test.extend([2] * x_test2.shape[0])
# x_train.extend(x_train2)
# x_test.extend(x_test2)


# Fault3 = np.nan_to_num(data1['Fault3'])
# num_samples3 = Fault3.shape[0]
# permuted_indices3 = np.random.permutation(num_samples3)
# split_point3 = int(num_samples3 * 0.9)
# x_train3 = Fault3[permuted_indices3[:split_point3]]
# y_train.extend([3] * x_train3.shape[0])
# x_test3 = Fault3[permuted_indices3[split_point3:]]
# y_test.extend([3] * x_test3.shape[0])
# x_train.extend(x_train3)
# x_test.extend(x_test3)

# Fault253 = np.nan_to_num(data25['Fault3'])
# num_samples253 = Fault253.shape[0]
# permuted_indices253 = np.random.permutation(num_samples253)
# split_point253 = int(num_samples253 * 0.9)
# x_train3 = Fault253[permuted_indices253[:split_point253]]
# y_train.extend([3] * x_train3.shape[0])
# x_test3 = Fault253[permuted_indices253[split_point253:]]
# y_test.extend([3] * x_test3.shape[0])
# x_train.extend(x_train3)
# x_test.extend(x_test3)

# Fault31 = np.nan_to_num(data11['Fault3'])
# num_samples31 = Fault31.shape[0]
# permuted_indices31 = np.random.permutation(num_samples31)
# split_point31 = int(num_samples31 * 0.9)
# x_train3 = Fault31[permuted_indices31[:split_point31]]
# y_train.extend([3] * x_train3.shape[0])
# x_test3 = Fault31[permuted_indices31[split_point31:]]
# y_test.extend([3] * x_test3.shape[0])
# x_train.extend(x_train3)
# x_test.extend(x_test3)



# Fault4 = np.nan_to_num(data1['Fault4'])
# num_samples4 = Fault4.shape[0]
# permuted_indices4 = np.random.permutation(num_samples4)
# split_point4 = int(num_samples4 * 0.8)
# x_train4 = Fault4[permuted_indices4[:split_point4]]
# y_train.extend([4] * x_train4.shape[0])
# x_test4 = Fault4[permuted_indices4[split_point4:]]
# y_test.extend([4] * x_test4.shape[0])
# x_train.extend(x_train4)
# x_test.extend(x_test4)


# Fault5 = np.nan_to_num(data1['Fault5'])
# num_samples5 = Fault5.shape[0]
# permuted_indices5 = np.random.permutation(num_samples5)
# split_point5 = int(num_samples5 * 0.8)
# x_train5 = Fault5[permuted_indices5[:split_point5]]
# y_train.extend([5] * x_train5.shape[0])
# x_test5 = Fault5[permuted_indices5[split_point5:]]
# y_test.extend([5] * x_test5.shape[0])
# x_train.extend(x_train5)
# x_test.extend(x_test5)


# Normal = np.nan_to_num(data2['NORMAL'])
# num_samples6 = Normal.shape[0]
# permuted_indices6 = np.random.permutation(num_samples6)
# split_point6 = int(num_samples6 * 0.9)
# x_train6 = Normal[permuted_indices6[:split_point6]]
# y_train.extend([0] * x_train6.shape[0])
# x_test6 = Normal[permuted_indices6[split_point6:]]
# y_test.extend([0] * x_test6.shape[0])
# x_train.extend(x_train6)
# x_test.extend(x_test6)


# Normal_t1 = x_train6[0:654]
# Normal_t2 = x_train6[654*1+1:654*2]
# Normal_t3 = x_train6[654*2+1:654*3]
# Normal_t4 = x_train6[654*3+1:654*4]
# Normal_t5 = x_train6[654*4+1:654*5]
# Normal_t6 = x_train6[654*5+1:654*6]

# Fault1_t2 = x_train1[]
# Fault1_t3 = x_train1[]
# Fault1_t4 = x_train1[]
# Fault1_t5 = x_train1[]
# Fault1_t6 = x_train1[]
# data_to_save = {
#     'x_train': x_train,
#     'x_test': x_test,
#     'y_train': y_train,
#     'y_test': y_test,
# }

# scipy.io.savemat('HRSs_d.mat', data_to_save)