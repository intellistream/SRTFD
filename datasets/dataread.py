# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:36:50 2024

@author: Zhao Dandan
"""
import datetime
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.io import savemat



def timestampIntToDateTime(inputInt):
        # Convert milliseconds to seconds and microseconds
        seconds = inputInt // 1000
        microseconds = (inputInt % 1000) * 1000
        # Create a datetime object with the precise timestamp
        dt = datetime.datetime.fromtimestamp(seconds) + datetime.timedelta(microseconds=microseconds)
        return dt
    
# 读取JSON文件
datatimee = []
Fault2 = []
Fault1 = []
Fault3 = []
with open('24PJ.json', 'r') as f:
    data = json.load(f)
    
    #Data1 = data[7]['电流数据']#338367:345567  244776:251976  323968:331168   219579:226778   298770:305970
    #259174:266374    256777:263940    205180:212380
    # datatimeint = Data1['time']
    # for i in range(len(datatimeint)):
    #     datatimee.append(timestampIntToDateTime(datatimeint[i]))
        
    x1 = data[0]['电流数据']['current']#[256777:263935]
    x2 = data[3]['电流数据']['current']#[256777:263935]
    x3 = data[4]['电流数据']['current']#[256777:263935]
    x4 = data[5]['电流数据']['current']#[256777:263935]
    x5 = data[6]['电流数据']['current']#[256777:263935]
    x6 = data[7]['电流数据']['current']#[256777:263935]
    x7 = data[1]['电流数据']['current']#[256777:263935]
    x8 = data[2]['电流数据']['current']#[256777:263935]
    # for i in range(10):
    #     Fault2.append(x1[338367+i*1000:338367+(i+1)*1000])
    #     Fault2.append(x2[219579+i*1000:219579+(i+1)*1000])
    #     Fault2.append(x3[298770+i*1000:298770+(i+1)*1000])
    #     Fault2.append(x4[259174+i*1000:259174+(i+1)*1000])
    #     Fault2.append(x5[256777+i*1000:256777+(i+1)*1000])
    #     Fault2.append(x6[338367+i*1000:338367+(i+1)*1000])
    #     Fault1.append(x7[205180+i*1000:205180+(i+1)*1000])
    #     Fault3.append(x8[323968+i*1000:323968+(i+1)*1000])
        
    for i in range(100):
        Fault2.append(x1[338367-100000+i*1000:338367-100000+(i+1)*1000])
        Fault2.append(x2[219579-100000+i*1000:219579-100000+(i+1)*1000])
        Fault2.append(x3[298770-100000+i*1000:298770-100000+(i+1)*1000])
        Fault2.append(x4[259174-100000+i*1000:259174-100000+(i+1)*1000])
        Fault2.append(x5[256777-100000+i*1000:256777-100000+(i+1)*1000])
        Fault2.append(x6[338367-100000+i*1000:338367-100000+(i+1)*1000])
        Fault1.append(x7[205180-100000+i*1000:205180-100000+(i+1)*1000])
        Fault3.append(x8[323968-100000+i*1000:323968-100000+(i+1)*1000])
        
    for j in range(50):
        Fault2.append(x1[338367+10000+i*1000:338367+10000+(i+1)*1000])
        Fault2.append(x2[219579+10000+i*1000:219579+10000+(i+1)*1000])
        Fault2.append(x3[298770+10000+i*1000:298770+10000+(i+1)*1000])
        Fault2.append(x4[259174+10000+i*1000:259174+10000+(i+1)*1000])
        Fault2.append(x5[256777+10000+i*1000:256777+10000+(i+1)*1000])
        Fault2.append(x6[338367+10000+i*1000:338367+10000+(i+1)*1000])
        Fault1.append(x7[205180+10000+i*1000:205180+10000+(i+1)*1000])
        Fault3.append(x8[323968+10000+i*1000:323968+10000+(i+1)*1000])
    
    Fault1 = np.array(Fault1)
    Fault2 = np.array(Fault2)
    Fault3 = np.array(Fault3)
    savemat('fault_duo.mat', {'Fault1': Fault1, 'Fault2': Fault2, 'Fault3':Fault3})
# # 绘制折线图
# plt.plot(x)
# plt.xlabel('time')
# plt.ylabel('current')
# plt.grid(True)  # 添加网格线
# plt.show()


