import numpy as np
import time
import torch

# 模拟50万个512维的人脸特征
def database():
    sign = np.random.randint(0, 2, (400000, 512))*2-1
    ff_N = np.random.random((400000, 512))*sign
    ff_N = ff_N/np.linalg.norm(ff_N, ord=2, axis=1, keepdims=True)
    return ff_N.astype(np.float32)

if __name__=='__main__':
    device = torch.device('cuda:0')

    ff_N = torch.from_numpy(database()).to(device)

    ff_1 = np.random.random((4,1,512)).astype(np.float32)

    ff_1 = torch.from_numpy(ff_1).to(device)

    start = time.time()

    for i in range(10):
        d = ff_1 - ff_N
        d = d*d
        d = torch.sum(d, axis=2)

        # 输出与拍到的人脸特征距离最小的数据库中人脸特征的序号
        n = torch.argmin(d, axis=1)
    print(n, time.time() - start)