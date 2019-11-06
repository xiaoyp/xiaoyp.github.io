---
title: 三维模型表示方法 相关论文总结
tags:
  - 3D reconstruction
category: Summary
date: 2019-11-06 14:57:32
---


## 体素（Voxel Grids）

### （2016 ECCV）Learning a predictable and generative vector representation for objects （[论文链接](https://arxiv.org/pdf/1603.08637.pdf)）

论文提出了TL-embedding Network，给出了一种对三维模型的表示，这一表示既能够用于三维模型的生成，也能够从二维图像中提取出来。网络结构分为两个部分，第一部分为自动编码器，得到三维模型的embeddings；第二部分为卷积神经网络，将二维图像提取特征信息映射到三维模型的embeddings上。但是在TL-embedding Network中，输入输出的体素模型大小为20*20*20，分辨率较低。

<!--more-->

### （2017 SIGGRAPH）GRASS: Generative Recursive Autoencoders for Shape Sturctures （[论文链接](https://arxiv.org/pdf/1705.02090.pdf)）

论文中将三维模型中的parts以层次结构的形式进行组织并对三维模型的structure进行编码，对于三维物体的结构，作者采用OBBs的形式进行表达；具体的几何细节利用体素表达。GRASS网络结构由三部分组成，第一部分为自编码器，其中的网络结构基于递归神经网络（RvNN）实现，达到将三维物体的不同part合并的目的；第二部分为GAN，训练这一部分使得网络能够生成新的三维模型；第三部分为自编码器，用于对模型几何结构的编码与表示。但是，采用RvNN的结构会存在很多不适合的层次结构，对这一部分的计算会占用大量的计算资源，因此GRASS很难处理高分辨率的模型。

### （2019 TOG） Global-to-Local Generative Model for 3D Shapes （[论文链接](https://vcc.tech/file/upload_file/image/research/att201809231254/G2L.pdf)）

论文通过GAN来生成三维模型，但考虑到GAN对于生成几何特征的表现并不好，作者通过GAN生成模型的大致结构以及每一个voxel对应的part的标签，再将这些信息按模型中的每一个part作为Part Refiner（PR）的输入，最终生成更精细的三维模型。其中，GAN的判别器被分为全局判别器与局部判别器，除此之外，作者提出了两种loss function保证GAN的训练效果。

### （2019 TOG） SAGNet: Sturcture-aware Generative Network for 3D-Shape Modeling （[论文链接](https://arxiv.org/pdf/1808.03981.pdf)）

论文提出需要在对三维模型编码的过程中考虑到几何特征与整体结构的关系，SAGNet网络中通过循环神经网络GRU以及神经网络中的注意力机制将结构与几何特征之间的关系加入到其各自的训练中去。考虑到几何特征与模型结构之间的关系，能够很好地实现对榫卯结构的生成。

## 点云（Point Clouds）

### （2018 CVPR）FoldingNet: Point Cloud Auto-Encoder via Deep Grid Deformation （[论文链接](https://arxiv.org/pdf/1712.07262.pdf)）

论文提出了一种基于折叠操作而建立的解码器用于将二维的网格点通过多次迭代操作最终折叠形成三维点云用于对三维模型的表示。模型中的编码器基于Graph，对点云中每个点的局部特征进行提取，最终得到512维的codeword，解码器将codeword复制后与二维网格点的位置相连，作为折叠操作的输入。

### （2019 SIGGRAPH）StructureNet: Hierarchical Graph Networks for 3D Shape Generation （[论文链接](https://arxiv.org/pdf/1908.00575.pdf)）

与GRASS方法相比，StructureNet采用图的形式描述模型的层次结构，首先通过模型的层次结构构建n叉树，再根据兄弟节点的关系（对称、连接等）连接对应的边。StructureNet采用基于图卷积的VAE对模型的Hierarchical Graph进行处理。

### （2019 ICCV） PU-GAN: a Point Cloud Upsampling Adversarial Network （[论文链接](https://arxiv.org/pdf/1907.10844.pdf)）

这篇论文通过上采样的方法将稀疏、有噪音、不均匀的点云模型转变为密集、完整、均匀的点云模型。整体网络结构采用GAN，生成器的输入为原始的三维点集合、输出为上采样之后的三维点集，生成器可被分为三个模块：特征提取、特征扩展、以及点集生成；判别器相当于对上采样后的点集进行特征提取，再在最后连接全连接层实现对生成效果的判断。除此之外，作者提出了一种新的loss function以确保生成更均匀的三维点集。

## 三角网格（Mesh）

### （2019 SIGGRAPH） SDM-NET: Deep Generative Network for Structured Deformable Mesh （[论文链接](https://arxiv.org/pdf/1908.04520.pdf)）

论文提出了一种能够生成结构化可变形三角网格模型的网络结构，SDM-NET为两层VAE结构，第一层被称为PartVAE，用于对模型的每一部分几何特征的学习，将每一部分视为bounding box通过变形得到；第二层被称为Structured Part VAE(SP-VAE)，同时学习各个部分间的结构以及每一部分的几何特征。

## 符号距离函数（Signed Distance Function）

### （2019 CVPR）DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation （[论文链接](http://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf)）

### （2019 NeurIPS） DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction （[论文链接](https://arxiv.org/pdf/1905.10711.pdf)）

DISN能够从二维的图像重建三维的模型，作者提出一种网络，首先根据二维图像学习相机参数，再根据这些参数将三维网格点映射到二维图像上，提取图像的全局特征以及映射点的局部特征，根据这些特征预测其SDF值，再根据SDF值重建三维模型。由于DISN采用了带符号距离函数而不是二值函数表示三维点相对于平面的位置，且在网络中添加了局部特征并修改了代价函数的权重，使得DISN与之前方法相比能够生成更精细的局部细节。

## Others

### （2017 ICCV） 3D-PRNN: Generating Shape Primitives with Recurrent Neural Networks （[论文链接](https://arxiv.org/pdf/1708.01648.pdf)）

### （2019 CVPR）Occupancy Networks: Learning 3D Reconstruction in Function Space （[论文链接](http://www.cvlibs.net/publications/Mescheder2019CVPR.pdf)）