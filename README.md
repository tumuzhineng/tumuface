# 人脸识别

## 代码说明

### 1. algorithms 文件夹包含人脸检测、人脸特征提取和人脸识别三个模块

### 2. 代码示例在 test 文件夹

### 3. 权重文件链接

链接：https://pan.baidu.com/s/1J8CU9rn9fEy1lzKNnxJXUA?pwd=298m 
提取码：298m 

将下载好的权重文件放在weights文件夹中

## 算法说明

### 1. 人脸检测
### 2. 人脸特征提取

算法处理人脸图像并生成一个512维的向量，该向量称为人脸特征向量(简称"特征")。这个特征位于一个512维且半径为1的超球体表面。

### 3. 人脸识别

人脸识别分为"一对一"人脸比对和"一对多"人脸识别；

"一对一"人脸比对，即比较两张人脸图像特征的相似度。一般计算这两个特征的欧氏距离，然后根据预先设置的阈值来判断这两个特征所对应的两张人脸图像是否属于同一个人；

"一对多"人脸识别，即将一张新的人脸图像的特征与人脸特征库中的每一个特征都进行比较，然后找出库中与其相似度最高的特征，若相似度符合阈值要求，则识别成功，否则未能从库中找到该人。"一对多"人脸识别包含多次"一对一"人脸比对。

所有特征均位于一个 512 维且半径为1的超球体表面，两个特征之间的欧式距离在[0,2]区间。为减少计算步骤，一般使用开平方之前为数据，区间为[0,4]。阈值一般设为1.16，若两个特征的平方距离小于1.16，则认为它们属于同一个人，否则属于不同的人。