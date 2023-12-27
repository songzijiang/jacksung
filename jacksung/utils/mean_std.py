import numpy as np


def mean_std_part2all(num_list, mean_list, std_list):
    # [(n,mean,std)]
    group_num = len(num_list)
    dim = mean_list[0].shape
    data_num = sum(num_list)

    global_mean = np.zeros(dim, dtype=np.float64)
    global_var = np.zeros(dim, dtype=np.float64)
    for i in range(group_num):
        global_mean += (num_list[i] / data_num * mean_list[i])
    for i in range(group_num):
        global_var += (num_list[i] / data_num * (std_list[i] ** 2 + (global_mean - mean_list[i]) ** 2))
    return global_mean, np.sqrt(global_var)


def cal_mean_std_one_loop(s, ss, count):
    mean_pixel = s / count
    mean_level = mean_pixel.mean((-2, -1))

    var_pixel = ss / count - np.square(mean_pixel)
    ss_level = ss.mean((-2, -1))
    var_level = ss_level / count - np.square(mean_level)

    std_level = np.sqrt(var_level)
    std_pixel = np.sqrt(var_pixel)

    return mean_pixel, std_pixel, mean_level, std_level


if __name__ == '__main__':
    d1 = np.random.rand(2, 3, 2, 2)
    d2 = np.random.rand(3, 3, 2, 2)
    d = np.concatenate([d1, d2])
    s1 = d1.sum((0,))
    ss1 = np.array([i ** 2 for i in d1]).sum((0,))

    s2 = d2.sum((0,))
    ss2 = np.array([i ** 2 for i in d2]).sum((0,))

    mean_pixel1, std_pixel1, mean_level1, std_level1 = cal_mean_std_one_loop(s1, ss1, 2)
    mean_pixel2, std_pixel2, mean_level2, std_level2 = cal_mean_std_one_loop(s2, ss2, 3)
    # print('*' * 40)
    # print(mean_pixel1, end='\n\n')
    # print(std_pixel1, end='\n\n')
    # print(mean_level1, end='\n\n')
    print(std_level1, end='\n\n')
    # print('*' * 40)
    # print(mean_pixel2, end='\n\n')
    # print(std_pixel2, end='\n\n')
    # print(mean_level2, end='\n\n')
    print(std_level2, end='\n\n')

    mean_pixel, std_pixel = mean_std_part2all([2, 3], [mean_pixel1, mean_pixel2], [std_pixel1, std_pixel2])
    mean_level, std_level = mean_std_part2all([2, 3], [mean_level1, mean_level2], [std_level1, std_level2])
    print('*' * 40)
    # print(mean_pixel, end='\n\n')
    # print(std_pixel, end='\n\n')
    # print(mean_level, end='\n\n')
    print(std_level, end='\n\n')
    print(d1.std((0, -2, -1)))
    print(d2.std((0, -2, -1)))
    print(d.std((0, -2, -1)))
