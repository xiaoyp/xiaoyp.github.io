---
title: GCNN 图卷积网络(二) 
date: 2019-10-30 15:55:42
tags:
    - Graph
category: Deep Learning
---

## 相关论文1： Convolutional Neural Networks on Graphs with Fast Localized Spetral Filtering

### Learning Fast Localized Spectral Filters

#### Spectral filtering of graph signals

在上文中，我们将传统的傅里叶变换推广到图的傅里叶变换，接下来我们考虑卷积操作。

<!--more-->

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