import numpy as np
from scipy.stats import sem
import scipy.stats as stats


def comp_avg(metric, t_coef, n_tasks):
    # shape: (num_run, num_task)
    end_metric = metric[:, -1, :]
    # mean of end task accuracies per run
    avg_metric_per_run = np.mean(end_metric, axis=1)
    avg_end_metric = (np.mean(avg_metric_per_run),
                      t_coef * sem(avg_metric_per_run))

    metric_per_run = np.mean((np.sum(np.tril(metric), axis=2) /
                              (np.arange(n_tasks) + 1)), axis=1)
    avg_metric = (np.mean(metric_per_run), t_coef * sem(metric_per_run))

    return avg_end_metric, avg_metric


def compute_performance(end_task_acc_arr, end_task_rec_arr, end_task_pre_arr, end_task_f1_arr, end_task_gmean_arr):
    """
    Given test accuracy results from multiple runs saved in end_task_acc_arr,
    compute the average accuracy, forgetting, and task accuracies as well as their confidence intervals.

    :param end_task_acc_arr:       (list) List of lists
    :param task_ids:                (list or tuple) Task ids to keep track of
    :return:                        (avg_end_acc, forgetting, avg_acc_task)
    """
    n_run, n_tasks = end_task_acc_arr.shape[:2]
    # t coefficient used to compute 95% CIs: mean +- t *
    t_coef = stats.t.ppf((1+0.95) / 2, n_run-1)

    # compute average test accuracy and CI
    # shape: (num_run, num_task)
    end_acc = end_task_acc_arr[:, -1, :]
    # mean of end task accuracies per run
    avg_acc_per_run = np.mean(end_acc, axis=1)
    avg_end_acc = (np.mean(avg_acc_per_run), t_coef * sem(avg_acc_per_run))

    # compute forgetting
    best_acc = np.max(end_task_acc_arr, axis=1)
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets, axis=1)
    avg_end_fgt = (np.mean(avg_fgt), t_coef * sem(avg_fgt))

    # compute ACC
    acc_per_run = np.mean((np.sum(np.tril(end_task_acc_arr), axis=2) /
                           (np.arange(n_tasks) + 1)), axis=1)
    avg_acc = (np.mean(acc_per_run), t_coef * sem(acc_per_run))

    avg_end_rec, avg_rec = comp_avg(end_task_rec_arr, t_coef, n_tasks)
    avg_end_pre, avg_pre = comp_avg(end_task_pre_arr, t_coef, n_tasks)
    avg_end_f1, avg_f1 = comp_avg(end_task_f1_arr, t_coef, n_tasks)
    avg_end_gmean, avg_gmean = comp_avg(end_task_gmean_arr, t_coef, n_tasks)

    # compute BWT+
    bwt_per_run = (np.sum(np.tril(end_task_acc_arr, -1), axis=(1, 2)) -
                   np.sum(np.diagonal(end_task_acc_arr, axis1=1, axis2=2) *
                          (np.arange(n_tasks - 1, 0, -1) - 1), axis=1)) / ((n_tasks - 1) * (n_tasks - 2) / 2)
    bwtp_per_run = np.maximum(bwt_per_run, 0)
    avg_bwtp = (np.mean(bwtp_per_run), t_coef * sem(bwtp_per_run))

    # compute FWT
    fwt_per_run = np.sum(np.triu(end_task_acc_arr, 1),
                         axis=(1, 2)) / (n_tasks * (n_tasks - 1) / 2)
    avg_fwt = (np.mean(fwt_per_run), t_coef * sem(fwt_per_run))
    return avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt, avg_end_rec, avg_rec, avg_end_pre, avg_pre, avg_end_f1, avg_f1, avg_end_gmean, avg_gmean


def single_run_avg_end_fgt(acc_array):
    best_acc = np.max(acc_array, axis=1)
    end_acc = acc_array[-1]
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets)
    return avg_fgt
