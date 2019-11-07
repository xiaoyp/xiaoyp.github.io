---
title: GCNN 图卷积网络
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
\hat{x} = U^Tx
$$

$$
x = U\hat{x}
$$

同在欧氏空间一样，这一转换支持基本操作的公式，如过滤。

#### Spectral filtering of graph signals

在上文中，我们将传统的傅里叶变换推广到图的傅里叶变换，接下来我们考虑卷积操作。

卷积定理：函数卷积的傅里叶变换是函数傅里叶变换的乘积，即对于函数$f(t)$与$h(t)$两者的卷积是其函数傅里叶变换乘积的逆变换：

$$
f * h = \mathcal{F}^{-1}[\hat{f}(\omega)\hat{h}(\omega)] = \frac{1}{2\Pi}\int\hat{f}(\omega)\hat{h}(\omega)e^{i\omega t}d\omega
$$

类比到Graph上并把傅里叶变换的定义带入，$x$与卷积核$h$可被计算：

$x$的傅里叶变换为$\hat{x} = U^Tx$，卷积核$h$的傅里叶变换写成对角矩阵的形式即为：

$$
\begin{bmatrix}
    \hat{h}(\lambda_1) & & \\
    & \ddots & \\
    & & \hat{h}(\lambda_n)
\end{bmatrix}
$$

其中，$\hat{h}(\lambda_l) = \sum^N_{i=1}h(i)u^*_l(i)$是根据需要设计的卷积核$h$在Graph上的傅里叶变换。这一矩阵也可以被写为$U^Th$。

Graph的信号x与卷积核h两者的傅里叶变换乘积即为$(U^Th)\odot(U^Tx)$（$\odot$表示Hadamard积，对于连个维度相同的向量、矩阵、张量进行对应位置的逐元素乘积运算），再乘以$U$求两者傅里叶变换乘积的逆变换，则求出卷积：

$$
(x * h)_G = U((U^Th)\odot(U^Tx))
$$

在图卷积网络中，我们令$g_{\theta}$表示网络中的卷积核，也就是说，Graph中每个节点的信号x可以被卷积核$g_{\theta}$所过滤并提取特征：

$$
y = g_{\theta}(L)x = g_{\theta}(U\Lambda U^T)x = Ug_{\theta}(\Lambda)U^Tx
$$

因此，一种非参数的卷积核，很自然地被定义为$g_{\theta}(\Lambda) = diag(\theta)$。

因此，卷积层被定义为$y_{output} = \sigma(Ug_{\theta}(\Lambda)U^Tx)$。根据这一定义，通过对卷积核参数初始化赋值后利用误差反向传播进行调增，x就是graph上对应于每个顶点的feature vector。但是，这一方法也存在弊端：首先，每一次前向传播，都需要计算$U, diag(\theta_l), U^T$三者的矩阵乘积，这需要比较大的计算复杂度，第二，they are not localized in space，第三，卷积核需要n个参数。

#### Polynomial parametrization for localized filters

为解决上述问题，论文中使用了polynomial filter：

$$
g_{\theta}(\Lambda) = \sum_{k=0}^{K-1}\theta_k\Lambda^k
$$

则每次卷积的输出为$y_{output} = \sigma(Ug_{\theta}(\Lambda)U^Tx)$

矩阵中对角元素被定义为:

$$
\hat{h}(\lambda_l) = \sum^{K-1}_{k=0}\theta_k\lambda_l^k
$$

式子中的参数$\theta \in \mathbb{R}^{K}$可被看作多项式的系数。其中，$(\theta_0, \theta_2, \cdots, \theta_{K-1})$是任意的参数。$K$就是卷积核的感受域，也就是说每次卷积会将中心顶点最邻近的K层邻居上的feature进行加权求和，权系数就是$\theta_k$。

经过上述对卷积核的改进，对于卷积后信号的计算可以重新被表示为：

$$
y_{output} = \sigma(U\sum_{j=0}^{K-1}\theta_j\Lambda^jU^T) = \sigma(\sum_{j=0}^{K-1}\theta_jL^jx)
$$

#### Recursive formulation for fast filtering

假设$x_1, x_2, \dots, x_n$是X的一个线性无关组，$c_1, c_2, \dots, c_n$为R中任意数，由元素$c_1x_1+c_2x_2+\dots+c_nx_n$组成的全体是X的一个子集，记为$Span\{x_1, x_2,\dots,x_n\}$。

采用新的卷积核来计算，需要计算$L^j$的值，但是这一计算仍然需要$O(n^2)$的复杂度，因此，论文作者采用Chebyshev多项式拟合卷积核的方法，来降低计算复杂度。卷积核$g_{\theta}(\Lambda)$可以利用截断的shifted Chebyshev多项式来逼近：

$$
g_{\theta}(\Lambda) = \sum_{k=0}^{K-1}\theta_kT_k(\widetilde{\Lambda})
$$

通过Chebyshev多项式进行逼近，经过卷积之后的结果可以表示为$y = g_{\theta}(L)x = \sum_{k=0}^{K-1}\theta_kT_k(\widetilde{L})x$，其中$T_k(\widetilde{L}) \in \mathbb{R}^{n \times n}$为Chebyshev多项式的第k项，$\widetilde{L} = 2L/\lambda_{max} - I_n$。

#### Recursive formulation for fast filtering

在上文中，我们得到经过filtering操作之后的信号输出为$y = g_{\theta}(L)x = \sum_{k=0}^{K-1}\theta_kT_k(\widetilde{L})x$，其中$\widetilde{L} = 2L/\lambda_{max} - I_n$。

为尽可能降低计算的复杂度，令$\bar{x}_k = T_k(\widetilde{L})x \in \mathbb{R}^n$。之后，采用递归公式$\bar{x}_k = 2\widetilde{L}\bar{x}_{k-1} - \bar{x}_{k-2}$（$\bar{x}_0=x, \bar{x}_1=\widetilde{L}x$）。通过这一递归公式，使得之前的矩阵乘法转变为矩阵与向量相乘，最终filtering的计算转变为$y = g_\theta(L)x = [\bar{x}_0, \dots, \bar{x}_{K-1}]\theta$，因此，最终计算的时间复杂度为$O(K|\mathcal{E}|)$。

### GCN代码

[参考链接](https://www.zhihu.com/question/54504471)