---
title: GCNN 图卷积网络(一) 
date: 2019-10-28 22:08:32
tags:
    - Graph
category: Deep Learning
---

## 传统卷积神经网络CNN

卷积在通常的形式中，是对两个实变函数的一种数学运算。在卷积网络的术语中，卷积的第一个参数通常叫做输入，第二个参数叫做核函数，输出有时被称为特征映射。在传统的卷积神经网络中，由于卷积操作具有稀疏交互、参数共享、等变表示的特征而被广泛应用。

<!--more-->

从本质上讲，CNN中卷积本质上是利用一个共享参数的过滤器，通过计算中心像素点以及相邻像素点的加权来实现对输入数据特征的提取。

## 图卷积网络GCNN

卷积神经网络在计算机视觉以及自然语言处理方面取得了巨大的成功，但是卷积神经网络的输入必须是Euclidean domain的数据，即要求输入数据有规则的空间结构。但是，依然有很多数据并不具备规则的空间结构，这些数据被称为Non Euclidean data。

### Graph

Graph Convolution Network中的Graph是指图论中的用顶点和边建立相关对应关系的拓扑图，具有两个基本的特征：

1. **每个节点都有自己的特征信息**

2. **图中的每个节点还具有结构信息**

### 图卷积算法

图卷积算子：

$$
h^{l+1}_i = \sigma(\sum_{j \in N_i}\frac{1}{c_{ij}}h^l_jw^l_{R_j})
$$

其中，设中心节点为*i*，$h^l_i$为节点*i*在第*l*层的特征表达；$c_{ij}$为归一化因子，比如取节点度的倒数；$N_i$为节点*i*的邻居，包含自身；$R_i$为节点$i$的类型；$w_{R_j}$表示$R_i$类型节点的变换权重参数。

总的来说，图卷积算法分为三步：

1. **发射：** 每一个节点将自身的特征信息经过变换后发送给邻居节点。

2. **接收：** 每个节点将邻居节点的特征聚集起来。

3. **变换：** 把前面的信息聚集之后做非线性变换。

## 相关论文1： Convolutional Neural Networks on Graphs with Fast Localized Spetral Filtering

&nbsp;&nbsp;&nbsp;&nbsp;[论文链接](https://arxiv.org/pdf/1606.09375.pdf)  &nbsp;&nbsp;&nbsp;&nbsp;  [代码链接](https://github.com/mdeff/cnn_graph)

### Introduction

社交网络中的用户数据、生物信息网络中的基因数据等数据都属于non-Euclidean data，这些数据能够通过图的结构来表示。谱图论就是其中一种可以用来研究图的强大的数学工具。对于图卷积网络来说，需要三个基本的步骤：

* （针对图设计的卷积核）The design of localized convolutional filters on graphs

* （将相似的节点集合起来）a graph coarsening procedure that groups together similar vertices

* （对图的池化操作）a graph pooling operation that trades spatial resolution for higher filter resolution

### background: 拉普拉斯算子和拉普拉斯矩阵

&nbsp; &nbsp;&nbsp; &nbsp; [拉普拉斯算子和拉普拉斯矩阵](https://zhuanlan.zhihu.com/p/67336297)

&nbsp; &nbsp;&nbsp; &nbsp; [拉普拉斯矩阵和拉普拉斯算子的关系](https://zhuanlan.zhihu.com/p/85287578)

### Learning Fast Localized Spectral Filters

#### 图傅里叶变换（Graph Fourier Transform）

对于无向图$\mathcal{G = (V, E, W)}$，其中$\mathcal{V}$为顶点集合，且$|\mathcal{V}| = n$，$\mathcal{E}$为边的集合，$\mathcal{W} \in \mathbb{R}^{n \times n}$为权重邻接矩阵。定义在图中节点上的信号$x : \mathcal{V} \to \mathbb{R}$被看作一个向量$x \in \mathbb{R}^{n \times n}$，其中，$x_i$表示在第*i*个节点上的*x*值。这一映射关系可以看作图的函数。

在谱图分析中，需要用到拉普拉斯算子，用于描述图中节点梯度的散度，其定义为:

$$
L = D - W \in \mathbb{R}^{n \times n}
$$

其中$D \in \mathbb{R}^{n \times n}$是对角矩阵，对角线上元素为$D_{ii} = \sum_{j}W_{ij}$，拉普拉斯算子的正则化定义为$L = I_n - D^{-1/2}WD^{-1/2}$，由于矩阵*L*是实对称半正定矩阵，因此，拉普拉斯矩阵的n个特征值都大于等于0。对拉普拉斯矩阵进行特征值分解：

$$
L = U \Lambda U^T
$$

其中，*U*为拉普拉斯矩阵的特征向量矩阵，是正交矩阵，$\Lambda$为拉普拉斯矩阵特征值组成的对角矩阵。

从传统的傅里叶变换出发，传统的傅里叶变换被定义为$F(\omega) = \mathcal{F}|f(t)| = \int{f(t)e^{-i \omega t}dt}$，即信号$f(t)$与基函数$e^{-i \omega t}$的积分。从数学上来看，$e^{-i \omega t}$是拉普拉斯算子的特征函数，$\omega$ 就与特征值有关。因此，仿照传统傅里叶变换的公式，定义Graph上的傅里叶变换为：

$$
F(\lambda_l) = \hat{x}(\lambda_l) = \sum_{i=1}^{N}x_iu_l^*(i)
$$

将其写成矩阵形式得到图的傅里叶变换及傅里叶逆变换：

$$
\hat{x} = U^Tx \\
x = U\hat{x}
$$

同在欧氏空间一样，这一转换支持基本操作的公式，如过滤。

[链接](https://www.zhihu.com/question/54504471)