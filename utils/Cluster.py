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
from scipy.stats import multivariate_normal
import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky

def kl_divergence_gaussian(mu1, cov1, mu2, cov2):
    """
    Calculate KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu1 : numpy array
        Mean of the first Gaussian distribution.
    cov1 : numpy array
        Covariance matrix of the first Gaussian distribution.
    mu2 : numpy array
        Mean of the second Gaussian distribution.
    cov2 : numpy array
        Covariance matrix of the second Gaussian distribution.
        
    Returns:
    kl_div : float
        KL divergence from P(mu1, cov1) to Q(mu2, cov2).
    """
    # Ensure covariance matrices are positive definite
    cov1 = cov1 + 1e-6 * np.eye(cov1.shape[0])
    cov2 = cov2 + 1e-6 * np.eye(cov2.shape[0])
    
    # Cholesky decomposition of covariance matrices
    cholesky_cov1 = cholesky(cov1, lower=True)
    cholesky_cov2 = cholesky(cov2, lower=True)
    
    # Compute the trace term
    trace_term = np.trace(np.linalg.solve(cov2, cov1))
    
    # Compute the quadratic term
    quadratic_term = np.dot(np.dot((mu2 - mu1), np.linalg.inv(cov2)), (mu2 - mu1))
    
    # Compute the dimensionality
    dimensionality = len(mu1)
    
    # Compute KL divergence
    kl_div = 0.5 * (trace_term + quadratic_term - dimensionality + np.log(np.linalg.det(cov2) / np.linalg.det(cov1)))
    
    return kl_div


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

def kl_divergence(p,q,smooth=1e-9):
    p_smooth = p + smooth
    q_smooth = q + smooth
    return np.sum(p_smooth * np.log(p_smooth / q_smooth))

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))

def find_zero_indexes(arr):
    indexes = []
    
    for i in range(len(arr)):
        if arr[i] == 0:
            indexes.append(i)
    
    return indexes

def KL_div(x_train, y_train, buffer, n_clusters):
    class_Buffer = np.unique(buffer.buffer_label) 
    x_train_clusterdata = x_train.reshape(x_train.shape[0],-1)
    mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=256, random_state=0)
    y_pred = mbk.fit_predict(x_train_clusterdata)

    #P_D = np.zeros(n_clusters) 
    #B_D = np.zeros(len(class_Buffer)) 
    KL_matrix = np.zeros((3, 3))
    KL = []
    Bdata = buffer.buffer_img.reshape(buffer.buffer_img.shape[0],-1).cpu().numpy()
    if np.all(Bdata == 0):
        x_train,y_train = x_train,y_train
    else:
        nn = 1
        for j in range(len(class_Buffer)):
            B_data = buffer.buffer_img[np.where(buffer.buffer_label == j)]
            B_Data_mean = np.mean(B_data.cpu().numpy(), axis = 0)
            B_Data_var = np.var(Bdata, axis = 0)
            cov1 = np.cov(Bdata, rowvar=False, bias=True)
            for i in range(n_clusters):
                cluster_points = x_train_clusterdata[y_pred == i]
                P_Data_mean = np.mean(cluster_points, axis = 0)
                P_Data_var =  np.var(cluster_points, axis = 0)
                cov2 = np.cov(cluster_points, rowvar=False, bias=True)
                kl_divergences  = (js_divergence(B_Data_mean, P_Data_mean)+ js_divergence(B_Data_var, P_Data_var))/2
                KL.append(kl_divergences)
        KL_A = np.array(KL, dtype=float)
        KL_A[np.isnan(KL_A)] = np.nanmax(KL_A)
        min_val = np.min(KL_A)
        max_val = np.max(KL_A)
        normalized_arr = (KL_A - min_val) / (max_val - min_val)
        zero_positions = find_zero_indexes(normalized_arr)
        set2 = set(zero_positions)
        classss = [i for i in range(n_clusters)]
        select_class= [x for x in classss if x not in set2]
        selected_indices = []
        for clss in range(len(select_class)):
            cls_indices = np.where(y_pred == select_class[clss])[0]
            selected_indices.extend(cls_indices)
        
        x_train = x_train[selected_indices]
        y_train = y_train[selected_indices]
    return  x_train, y_train