'''
人脸比对服务平台
'''
import sys
sys.path.append('.')
import torch
from algorithms.face_feature_extract import FaceFeatureExtractor

if __name__ == '__main__':

    # 选择运行模型的设备
    device = torch.device('cuda:0')

    # 加载模型：人脸特征提取器 face feature extractor
    ff_extractor = FaceFeatureExtractor(pretrained='weights/20180402-114759-vggface2.pt').eval().to(device)
    
    dummy_input = torch.rand(1, 3, 160, 160).to(device)
    
    # 生成独立运行的模型权重文件
    with torch.no_grad(): 
        jit_model = torch.jit.trace(ff_extractor, dummy_input)

    jit_model.save('weights/face_feature_extractor.pt')
    