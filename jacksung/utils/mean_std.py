import numpy as np


def mean_std_part2all(num_list, mean_list, var_list):
    # [(n,mean,std)]
    group_num = len(mean_list)
    dim = mean_list[0].shape
    data_num = sum(num_list)

    global_mean = np.zeros(dim, dtype=np.float64)
    global_var = np.zeros(dim, dtype=np.float64)
    for i in range(group_num):
        global_mean += (num_list[i] / data_num * mean_list[i])
    for i in range(group_num):
        global_var += (num_list[i] / data_num * (var_list[i] + (global_mean - mean_list[i]) ** 2))
    return global_mean, np.sqrt(global_var)


def cal_mean_std_one_loop(s, ss, count):
    mean_pixel = s / count
    mean_level = mean_pixel.mean((2, 3))

    var_pixel = ss / count - np.square(mean_pixel)
    ss_level = ss.mean((2, 3))
    var_level = ss_level / count - np.square(mean_level)

    std_level = np.sqrt(var_level)
    std_pixel = np.sqrt(var_pixel)

    return mean_pixel, std_pixel, mean_level, std_level
