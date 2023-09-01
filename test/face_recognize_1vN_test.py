'''
人脸比对服务平台
'''
import sys
sys.path.append('.')
import time
import numpy as np
import torch
import cv2
from algorithms.face_feature_extract import FaceFeatureExtractor
from algorithms.face_feature_extract import pre_process
from algorithms.face_recognize import one_to_N, one_to_N_GPU
import time

# 数据库函数：从数据库中读取全部 N 条人脸特征。设 N = 50 0000，数据量为 50 0000*512*4 = 10 2400 0000 Bytes = 976.56 MB
def database():
    sign = np.random.randint(0, 2, (500000, 512))*2-1
    ff_N = np.random.random((500000, 512))*sign
    ff_N = ff_N/np.linalg.norm(ff_N, ord=2, axis=1, keepdims=True)
    return ff_N

if __name__ == '__main__':

    # 选择运行模型的设备
    device = torch.device('cuda:0')

    #ff_extractor = torch.jit.load('weights/face_feature_extractor.pt').to(device)

    # 加载模型：人脸特征提取器 face feature extractor
    ff_extractor = FaceFeatureExtractor(pretrained='weights/20180402-114759-vggface2.pt').eval().to(device)

    # 预热
    face_image = cv2.imread('data/test_images_aligned/kate_siegel/0.png')
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    face_image = pre_process(face_image).to(device)

    with torch.no_grad():
        ff_1 = ff_extractor(face_image).cpu().numpy()[0]

    # CPU 比对速度
    ff_N = database()

    start = time.time()

    face_image = cv2.imread('data/test_images_aligned/angelina_jolie/0.png')
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    face_image = pre_process(face_image).to(device)

    with torch.no_grad():
        ff_1 = ff_extractor(face_image).cpu().numpy()[0]

    n = one_to_N(ff_1, ff_N, t=1.16)
    if n == -1:
        print('未识别到！')
    else:
        print('识别到人员编号：', n)

    print('耗时：', time.time()-start, 's')
    

    # GPU 比对速度
    ff_N = torch.from_numpy(database()).to(device)

    start = time.time()

    face_image = cv2.imread('data/test_images_aligned/bradley_cooper/0.png')
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    face_image = pre_process(face_image).to(device)
    with torch.no_grad():
        ff_1 = ff_extractor(face_image)

    n = one_to_N_GPU(ff_1, ff_N, t=1.16)

    if n == -1:
        print('未识别到！')
    else:
        print('识别到人员编号：', n)

    print('耗时：', time.time()-start, 's')