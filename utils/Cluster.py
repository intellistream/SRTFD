# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:47:31 2024

@author: Zhao Dandan
"""

from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.cluster import MeanShift, MiniBatchKMeans
import numpy as np
from utils.buffer.buffer import Buffer

def Cluster(x_train, y_train, sample_rate):
    x_train_clusterdata = x_train.reshape(x_train.shape[0],-1)
    # meanshit = MeanShift()
    # meanshit.fit(x_train_clusterdata)
    # y_pred = meanshit.labels_
    mbk = MiniBatchKMeans(n_clusters=5, batch_size=128, random_state=0)
    y_pred = mbk.fit_predict(x_train_clusterdata)
    #y_pred = SpectralClustering().fit_predict(x_train_clusterdata)
    unique_classes = np.unique(y_pred)  
    
    selected_indices = []
    for clss in unique_classes:
        cls_indices = np.where(y_pred == clss)[0]
        np.random.shuffle(cls_indices)
        half_count = int(len(cls_indices) * sample_rate)
        selected_indices.extend(cls_indices[:half_count])
    
    x_train = x_train[selected_indices]
    y_train = y_train[selected_indices]
    return x_train, y_train

def kl_divergence(p,q):
    return np.sum(p * np.log(p / q))



def KL_div(x_train, y_train, n_clusters, buffer):
    x_train_clusterdata = x_train.reshape(x_train.shape[0],-1)
    mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=256, random_state=0)
    y_pred = mbk.fit_predict(x_train_clusterdata)

    cluster_centers = mbk.cluster_centers_
    cluster_variances = np.zeros(n_clusters)
    cluster_mean = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_points = x_train_clusterdata[y_pred == i]
        cluster_variances[i] = np.mean(np.var(cluster_points, axis = 0))
        cluster_mean[i] = np.mean(np.mean(cluster_points, axis = 0))
        
    kl_divergences = np.zeros(n_clusters)
    
    if np.all(buffer.buffer_img.reshape(buffer.buffer_img.shape[0],-1).cpu().numpy()==0):
        x_train, y_train = x_train, y_train
    else:
        Bdata = buffer.buffer_img.reshape(buffer.buffer_img.shape[0],-1).cpu().numpy()
        Bdata_mean = np.mean(Bdata, axis = 0)
        Bdata_variances = np.var(Bdata, axis = 0)
        for i in range(n_clusters):
            kl_divergences[i] = kl_divergence(Bdata_mean, cluster_mean[i]) + kl_divergence(Bdata_variances, cluster_variances[i]) 
        kl_divergences /= cluster_variances
        top_kl_indices = np.argsort(kl_divergences)[::-1][:8]#改成8
        
        selected_indices = []
        for clss in range(len(top_kl_indices)):
            cls_indices = np.where(y_pred == top_kl_indices[clss])[0]
            selected_indices.extend(cls_indices)
        
        x_train = x_train[selected_indices]
        y_train = y_train[selected_indices]
    return  x_train, y_train