import torch
import numpy as np

# 一对一人脸比对函数，计算两个特征的相似性，即欧几里得距离(一般用平方代替)
def one_to_one(ff_1, ff_2, t):
    d = np.sum(np.square(ff_1 - ff_2))
    if d>t:
        return -1
        
    return 1

# 一对多人脸识别(比对)函数GPU版
def one_to_N_GPU(ff_1, ff_N, t):

    # 计算拍到的人脸特征和数据库里人脸特征的欧几里得距离 d
    d = ff_1-ff_N
    d = d*d
    d = torch.sum(d, axis=1)

    # 输出与拍到的人脸特征距离最小的数据库中人脸特征的序号
    n = torch.argmin(d)

    # 比较 d[n] 和阈值 t 的大小，如果小于 t，则拍到的人脸为数据库中第 n 个人，否则无法识别此人。
    min_dist = d[n].item()
    if min_dist>t:
        return -1
        
    return n.item()

# 一对多人脸识别(比对)函数
def one_to_N(ff_1, ff_N, t):
    # 计算拍到的人脸特征和数据库里人脸特征的欧几里得距离 d
    d = ff_1-ff_N
    d = d*d
    d = np.sum(d, axis=1)

    # 输出与拍到的人脸特征距离最小的数据库中人脸特征的序号
    n = np.argmin(d)

    # 比较 d[n] 和阈值 t 的大小，如果小于 t，则拍到的人脸为数据库中第 n 个人，否则无法识别此人。
    if d[n]>t:
        return -1
    
    return n