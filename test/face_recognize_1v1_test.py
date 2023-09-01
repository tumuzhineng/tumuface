import sys
sys.path.append('.')
import torch
import cv2
import numpy as np
from recognition.face_feature_extract import FaceFeatureExtractor, pre_process
from recognition.face_recognize import one_to_one
import time


if __name__ == '__main__':

    # 选择运行模型的设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载模型：人脸特征提取器 face feature extractor
    ff_extractor = FaceFeatureExtractor(pretrained='weights/20180402-114759-vggface2.pt').eval().to(device)

    # 读取两张人脸图片，RBG格式，每张图片尺寸是 160x160x3，转换成 1x3x160x160 float32，像素值归一化到 (-1,1) 区间
    face_image_1 = cv2.imread('data/test_images_aligned/angelina_jolie/0.png')
    face_image_1 = cv2.cvtColor(face_image_1, cv2.COLOR_BGR2RGB)
    face_image_2 = cv2.imread('data/test_images_aligned/bradley_cooper/0.png')
    face_image_2 = cv2.cvtColor(face_image_2, cv2.COLOR_BGR2RGB)

    #face_image = face_image[::-1]

    face_image_1 = pre_process(face_image_1).to(device)
    face_image_2 = pre_process(face_image_2).to(device)

    # 提取人脸特征
    with torch.no_grad():
        features_1 = ff_extractor(face_image_1).cpu().numpy()
        features_2 = ff_extractor(face_image_2).cpu().numpy()

    result = one_to_one(features_1, features_2, 1.16)

    if result == 1:
        print('验证通过')
    if result == -1:
        print('验证失败')
    
    '''
    # 时间测试
    print('Start')
    start = time.time()
    for i in range(1000):
        image = torch.rand((1,3,160,160)).to(device)
        features = ff_extractor(image).detach().cpu()
    print((time.time()-start)/1000)
    '''
