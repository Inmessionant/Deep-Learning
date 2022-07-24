



## Anaconda




```
# 清理缓存
conda clean -a

# 安装requirements里面的版本
conda install --yes --file requirements.txt

# 测试cuda是否可用
import torch
import torchvision
print(torch.cuda.is_available())
print(torch.version.cuda)

# 删除conda环境
conda remove -n name --all

# conda换源记得去掉default，添加pytorch
```



```
# conda 创建环境 + 装cuda + PyTorch

conda create -n name python=3.8
conda install cudatoolkit=10.1
conda install cudnn
使用pytorch官网的pip/conda命令装torch和torchvision
```



------



## Python



- [ ] Effective Python：编写高质量Python代码的90个有效方法

- [ ] 利用Python进行数据分析

- [ ] 流畅的Python

- [ ] [Numpy QuickStart](https://numpy.org/doc/stable/user/quickstart.html)



- [x] Python小技巧：https://space.bilibili.com/8032966/favlist?fid=413538&ftype=collect&ctype=21

  

------



## C++



- [x] C++ Primer
- [ ] Effective C++：改善程序与设计的55个具体做法



- [x] C++细节：https://space.bilibili.com/8032966/favlist?fid=573867&ftype=collect&ctype=21



------



## 操作系统



- [ ] 陈海波 - 现代操作系统：原理与实现
- [ ] Operating Systems:Three Easy Pieces  操作系统导论




- [ ] 蒋炎岩 - 南京大学操作系统



------



## 机器学习



- [ ] 百面机器学习
- [ ] 李航 - 统计学习方法第二版
- [ ] Pattern Recognition and Machine Learning
- [ ] Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow



- [ ] 林轩田 - 机器学习基石和技法
- [ ] 李宏毅 - Machine Learning（2019机器学习, 2021深度学习）
- [ ] 十分钟机器学习：https://www.bilibili.com/video/BV1No4y1o7ac?spm_id_from=333.999.0.0&vd_source=76d72bc0e66dc04e5ace0a2a6b2cd8a0



------



## 深度学习



- [ ] Hulu - 百面深度学习
- [ ] Goodfellow Bengio - Deep Learning
- [ ] 邱锡鹏 - 神经网络与深度学习



- [x] 李沐 - 动手学习深度学习（PyTorch）v2：https://courses.d2l.ai/zh-v2/

- [ ] 深度学习 - 纽约大学：https://www.bilibili.com/video/BV1Lb4y1X77t?spm_id_from=333.337.search-card.all.click&vd_source=76d72bc0e66dc04e5ace0a2a6b2cd8a0



------



## 2014



- [x] **GoogleNet(Inception):** Going deeper with convolutions

  https://arxiv.org/pdf/1409.4842.pdf

  https://zhuanlan.zhihu.com/p/32702031



------



## 2015



- [x] **Batch Normalization:** Accelerating Deep Network Training by Reducing Internal Covariate Shift

  https://arxiv.org/abs/1502.03167

  https://zhuanlan.zhihu.com/p/93643523



------



## 2016



- [x] **YOLO v1**: You Only Look Once: Unified, Real-Time Object Detection

  https://arxiv.org/abs/1506.02640

  https://zhuanlan.zhihu.com/p/32525231

  

- [x] **SqueezeNet** : AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size

  https://zhuanlan.zhihu.com/p/49465950

  https://arxiv.org/abs/1602.07360

  

- [x] **Faster RCNN**: Towards Real-Time Object Detection with Region Proposal Networks

  https://arxiv.org/pdf/1506.01497.pdf;

  https://zhuanlan.zhihu.com/p/31426458

  https://zhuanlan.zhihu.com/p/145842317

  

- [x] **YOLO v2**: YOLO9000: Better, Faster, Stronger

  https://zhuanlan.zhihu.com/p/74540100

  https://arxiv.org/abs/1612.08242

  

- [x] **ResNet**: Deep Residual Learning for Image Recognition

  https://arxiv.org/abs/1512.03385

  https://zhuanlan.zhihu.com/p/77794592

  https://zhuanlan.zhihu.com/p/31852747



- [ ] **OHEM**：Training Region-based Object Detectors with Online Hard Example Mining

  https://arxiv.org/pdf/1604.03540.pdf

  https://zhuanlan.zhihu.com/p/58162337



------



## 2017



- [ ] **Deformable ConvNet v1:** Deformable Convolutional Networks

  

- [x] **RetinaNet**: Focal Loss for Dense Object Detection

  https://zhuanlan.zhihu.com/p/133317452

  https://arxiv.org/abs/1708.02002

  

- [x] **SENet**: Squeeze-and-Excitation Networks

  https://arxiv.org/abs/1709.01507

  https://zhuanlan.zhihu.com/p/65459972

  

- [x] **ShuffleNet**: An Extremely Efficient Convolutional Neural Network for Mobile Devices 

  https://arxiv.org/pdf/1707.01083.pdf

  https://zhuanlan.zhihu.com/p/32304419

  

- [x] **ResNeXt**: Aggregated Residual Transformations for Deep Neural Networks

  https://arxiv.org/abs/1611.05431

  https://zhuanlan.zhihu.com/p/51075096

  https://zhuanlan.zhihu.com/p/78019001

  

- [x] **WideResNet**: Wide Residual Networks

  https://arxiv.org/abs/1605.07146

  

- [x] **DenseNet**: Densely Connected Convolutional Networks

  https://arxiv.org/abs/1608.06993

  https://zhuanlan.zhihu.com/p/37189203

  

- [x] **MobileNets**: Efficient Convolutional Neural Networks for Mobile Vision Applications

  https://arxiv.org/abs/1704.04861

  https://zhuanlan.zhihu.com/p/70703846

  

- [x] **FPN**：Feature Pyramid Networks for Object Detection

  https://arxiv.org/abs/1612.03144

  https://zhuanlan.zhihu.com/p/78160468

  https://zhuanlan.zhihu.com/p/60340636



- [x] An overview of **gradient descent optimization algorithms**

  https://ruder.io/optimizing-gradient-descent/index.html



------



## 2018



- [x] **CornerNet**: Detecting Objects as Paired Keypoints（Anchor free）

  https://arxiv.org/abs/1808.01244

  https://zhuanlan.zhihu.com/p/195517472

  

- [x] **MobileNetV2**: Inverted Residuals and Linear Bottlenecks

  https://arxiv.org/abs/1801.04381

  https://zhuanlan.zhihu.com/p/70703846

  

- [x] **ShuffleNet v2:** Practical Guidelines for Efficient CNN Architecture Design

  https://arxiv.org/abs/1807.11164

  https://zhuanlan.zhihu.com/p/48261931

  

- [x] **PANet**: Path Aggregation Network for Instance Segmentation

  https://arxiv.org/abs/1803.01534

  https://zhuanlan.zhihu.com/p/110204563

  https://zhuanlan.zhihu.com/p/34472945

  

- [x] **YOLO v3** : An Incremental Improvement

  https://arxiv.org/pdf/1804.02767.pdf

  https://zhuanlan.zhihu.com/p/143747206

  https://zhuanlan.zhihu.com/p/76802514

  

------



## 2019



- [ ] **Deformable ConvNet v2:** More Deformable, Better Results

  

- [x] **ACNet:** Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks

  

- [x] **ThunderNet:** Towards Real-time Generic Object Detection

  

- [x] **SKNet**: Selective Kernel Networks

  https://arxiv.org/abs/1903.06586

  https://zhuanlan.zhihu.com/p/59690223

  

- [x] **Res2Net:** A New Multi-scale Backbone Architecture

  

- [ ] **CenterNet:** Objects as Points

  

- [x] **EfficientNet**: Rethinking Model Scaling for Convolutional Neural Networks

  https://arxiv.org/pdf/1905.11946.pdf

  

- [x] Searching for **MobileNetV3**

  https://arxiv.org/abs/1905.02244

  https://zhuanlan.zhihu.com/p/70703846

  

- [ ] **RepPoints**: Point Set Representation for Object Detection



- [x] Bag of Tricks for Image Classification with Convolutional Neural Networks

  https://arxiv.org/pdf/1812.01187.pdf

  https://zhuanlan.zhihu.com/p/51870052

  

- [x] **FCOS:** Fully Convolutional One-Stage Object Detection（Anchor free）

  https://arxiv.org/pdf/2006.09214.pdf（2020）

  https://zhuanlan.zhihu.com/p/62869137

  https://mp.weixin.qq.com/s/2YiLmypIMuJQledtE-Utfw
  
  


------



## 2020



- [x] **DETR**: End-to-End Object Detection with Transformers（Transformer）

  https://arxiv.org/abs/2005.12872

  https://zhuanlan.zhihu.com/p/267156624

  

- [x] **ViT：**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

  https://arxiv.org/abs/2010.11929

  https://zhuanlan.zhihu.com/p/356155277

  

- [ ] **RepPoints v2:** Verification Meets Regression for Object Detection

  

- [x] **YOLO v4：** Optimal Speed and Accuracy of Object Detection

  https://zhuanlan.zhihu.com/p/143747206

  https://cloud.tencent.com/developer/article/1620195

  

- [x] **CSPNet:** A New Backbone that can Enhance Learning Capability of CNN 

  https://zhuanlan.zhihu.com/p/124838243

  

- [x] **U2-Net:** Going deeper with nested U-structure for salient object detection

  https://arxiv.org/pdf/2005.09007.pdf

  

- [x] **BBN**: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition

  http://www.weixiushen.com/publication/cvpr20_BBN.pdf

  https://zhuanlan.zhihu.com/p/123876769

  

- [x] **EfficientDet**: Scalable and Efficient Object Detection

  https://arxiv.org/pdf/1911.09070.pdf

  

- [ ] **D2Det**: Towards High Quality Object Detection and Instance Segmentation

  

- [x] **GhostNet**: More Features from Cheap Operations

  https://arxiv.org/abs/1911.11907

  https://zhuanlan.zhihu.com/p/109325275

  https://mp.weixin.qq.com/s/znibNmIO5Yjwm4hytAsPpg

  

- [ ] **SOLO**: Segmenting Objects by Locations

  https://arxiv.org/abs/1912.04488

  

- [ ] **SOLOv2**: Dynamic and Fast Instance Segmentation

  https://arxiv.org/abs/2003.10152

  

- [x] **PP-YOLO**: An Effective and Efficient Implementation of Object Detector

  https://arxiv.org/pdf/2007.12099.pdf

  https://zhuanlan.zhihu.com/p/164704942




------



## 2021



- [x] **RepVGG**: Making VGG-style ConvNets Great Again

  https://arxiv.org/pdf/2101.03697.pdf

  https://mp.weixin.qq.com/s/cQsfQ5Ea0utOAp3haig4Gw

  

- [x] **YOLOF**:You Only Look One-level Feature

  https://arxiv.org/abs/2103.09460

  https://blog.csdn.net/Q1u1NG/article/details/115168451

  

- [x] Bag of Tricks for Long-Tailed Visual Recognition with Deep Convolutional Neural Networks

  https://cs.nju.edu.cn/wujx/paper/AAAI2021_Tricks.pdf

  https://mp.weixin.qq.com/s/JC7h6x0PczfLDkV9zX_1HQ

  


- [ ] Revisiting ResNets: Improved Training and Scaling Strategies

  https://arxiv.org/pdf/2103.07579.pdf

  https://mp.weixin.qq.com/s/bt8kLN1a5BrUu8zaoJcXeA

  

- [x] **Swin Transformer**: Hierarchical Vision Transformer using Shifted Windows（Transformer）

  https://arxiv.org/abs/2103.14030



- [ ] Towards Open World Object Detection

  https://arxiv.org/abs/2103.02603



- [x] **YOLOX**: Exceeding YOLO Series in 2021

  https://arxiv.org/abs/2107.08430

  

- [x] **PP-YOLOv2**: A Practical Object Detector

  https://arxiv.org/abs/2104.10419

  https://zhuanlan.zhihu.com/p/367312049

  


------



## 2022



- [x] **ConvNeXt**：A ConvNet for the 2020s

  https://arxiv.org/abs/2201.03545

  https://zhuanlan.zhihu.com/p/458016349



- [ ] 
