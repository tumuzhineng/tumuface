import sys
sys.path.append('.')
import torch
from algorithms.face_detect import FaceDetector
from PIL import Image
import numpy as np

if __name__ == '__main__':
    # 选择运行模型的设备
    device = torch.device('cuda:0')

    face_detector = FaceDetector(pretrained='weights/',
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    image = Image.open('data/test_images/angelina_jolie/1.jpg')
    face_image, prob = face_detector(image, return_prob=True)
    print(prob)