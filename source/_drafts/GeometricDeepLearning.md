---
title: Geometry deep learning
date: 2019-12-27 21:23:14
tags:
category: Deep Learning
---

## 参考论文

**[Geometric deep learning: going beyond Euclidean data](https://arxiv.org/pdf/1611.08097.pdf)**

<!--more-->

## 符号

* $\mathbb{R}^m$：m维欧氏空间
* $a, \textbf{a}, \textbf{A}$：标量，向量，矩阵
* $\bar{a}$：$a$的共轭
* $\Omega, x$：任意域，域中的坐标
* $f \in L^2(\Omega)$：$\Omega$域内的平方可积函数
* $\delta_{x'}(x), \delta_{ij}$：克罗内克符号
* $\{f_i, y_i\}_{i \in \mathcal{I}}$：训练集
* $\mathcal{T}_v$：变换算子
* $\tau, \mathcal{L}_{\tau}$：变形域，变形算子
* $\hat{f}$：$f$的傅里叶变换
* $f \star g$：$f$与$g$的卷积
* $\mathcal{X}, T\mathcal{X}, T_x\mathcal{X}$：流形，流形的切平面束，在$x$处的切平面束
* $<\cdot, \cdot>_{T\mathcal{X}}$：黎曼度量
* $f \in L^2(\mathcal{X})$：流形$\mathcal{X}$的标量域
* $F \in L^2(T\mathcal{X})$：流形$\mathcal{X}$的切向量场
* $A^*$：$A$的伴随矩阵
* $\nabla, div, \Delta$：梯度，散度，拉普拉斯算子
* $\mathcal{V,E,F}$：图的顶点与边，网格的面
* $w_{ij}, \textbf{W}$：图的权值矩阵
* $f \in L^2(\mathcal{V})$：图顶点上的函数
* $F \in L^2(\mathcal{E})$：图边上的函数
* $\phi_i, \lambda_i$：拉普拉斯特征向量，特征值
* $h_t(\cdot, \cdot)$：热核
* $\Phi_k$：前$k$个特征向量的矩阵
* $\Lambda_k$：前$k$个拉普拉斯特征值的对角矩阵
* $\xi$：点的非线性函数（ReLU）
* $\gamma_{l,l'}(x), \Gamma_{l,l'}$：在空间和谱域的卷积过滤器

## 在欧氏空间的深度学习模型

### Geometric proiors

考虑d维的欧氏空间$\Omega = [0,1]^d \subset \mathbb{R}^d$，在这一欧氏空间上，定义平方可积函数$f \in L^2(\Omega)$（例如，在图像分析领域，图像可被看作是平方单位$\Omega = [0,1]^2$内的函数）。考虑通用的监督学习的表示方式，在训练集$\{ f_i \in L^2(\Omega), y_i = y(f_i)\}_{i \in \mathcal{I}}$中存在一未知函数$y: L^2(\Omega) \to \mathcal{Y}$。目标空间$\mathcal{Y}$根据不同的任务而不同。

对于未知函数$y$，会对其添加一些重要的先验假设，这些先验假设能够被卷积神经网络结构有效利用。

*Stationarity（平稳性）：*

$$
\mathcal{T_v}f(x) = f(x-v), \hspace{1cm} x,v \in \Omega
$$

$\mathcal{T_v}$是作用在函数$f \in L^2(\Omega)$上的转移算子。第一个假设是，函数$y$关于转移操作是不变或者等价的。对于不变性，假设$y(\mathcal{T}_vf) = y(f)$，这一假设属于物体分类任务的一个典型情况。对于等价性，我们假设$y(\mathcal{T_v}f) = \mathcal{T_v}y(f)$，当模型的输出是一个空间时，转移操作可以作用在输出上（例如语义分割，动作预测）。

*Local deformations and scale separation：*

同样地，对于变形算子$\mathcal{L}_{\tau}$，其中$\tau : \Omega \to \Omega$为平滑的向量场，作用在$L^2(\Omega)$上，公式为$\mathcal{L}_{\tau}f(x) = f(x - \tau(x))$。变形可被用来描述局部转移，视角的变换，旋转以及频率变化。

对于具有translation invariance的任务来说，有$|y(\mathcal{L}_{\tau}f) - y(f)| \approx ||\nabla_\tau||$。其中$||\nabla_\tau||$反映了给定变形域的光滑程度，即对于输入的微小变化，预测的结果不会发生大的变化。而对于具有translation equivariant的任务来说，有$|y(\mathcal{L}_{\tau}f)-\mathcal{L}_{\tau}y(f)| \approx ||\nabla_{\tau}||$，这一假设相较不变性假设来说更强。

根据translation invariance，通过下采样的局部过滤器，可以在较低的空间分辨率下提取到足够的统计信息。因此，对于一个图像中大范围的依赖关系可被切分为不同分辨率下的局部关系的相互作用，因此，构建层次模型，空间分辨率逐层递减。

### Convolutional neural networks

CNN包括卷积层$\textbf{g} = C_{\Gamma}(\textbf{f})$，输入为p维向量$\textbf{f}(x) = (f_1(x), \cdots, f_p(x))$，作用在一系列的过滤器$\Gamma = (\gamma_{l,l'}), l=1,\cdots, q,l'=1, \cdots, p$上，再经过非线性函数$\xi$，有：

$$
g_l(x) = \xi(\sum_{l'=1}^{p}(f_{l'} \star \gamma_{l,l'})(x))
$$

输出维q维向量$\textbf{g}(x) = (g_1(x), \cdots, g_q(x))$，表示输入的特征映射，标准卷积操作的公式为：

$$
(f \star \gamma)(x) = \int_{\Omega}f(x-x')\gamma(x')dx'
$$

池化层被定义为$\textbf{g} = P(\textbf{f})$，公式为：

$$
g_l(x) = P(\{f_l(x'):x' \in \mathcal{N}(x)\}), l=1, \cdots, q
$$

其中，$\mathcal{N}(x) \subset \Omega$是点$x$的邻域，函数$P$是满足排列不变性的函数。

## The Geometry of manifolds and graphs

对于非欧氏空间，两种主要的结构包括流形以及图。虽然这两种结构产生于不同的数学领域，但这两种结构间也具有一些类似的特征。

### 流形

通常来说，流形在局部空间内可被看作是欧氏空间。一个可微的$d$维流形$\mathcal{X}$是一个拓扑空间，其中的每一个点$x$的邻域与$d$维的欧氏空间同胚，这一欧氏空间被称为流形的切平面，记作$T_x\mathcal{X}$。所有点的切平面集合被称为切平面束，记作$T\mathcal{X}$。在这基础上定义切平面间的内积$<\cdot, \cdot>_{T_x\mathcal{X}}: T_x\mathcal{X} \times T_x\mathcal{X} \to \mathbb{R}$。这一内积也被称为黎曼度量。具有黎曼度量的流形也被称为黎曼曲面。

**Nash Embedding Theorem: Every smooth Riemannian manifold can be sommthly isometrically embedded into some Euclidean space.**

因此，二维流形被嵌入到三维欧氏空间中，被用于图形学领域来表示三维模型的曲面。

### 流形上的微积分

下一步是考虑定义在流形上的函数。函数可被分为两类：第一类是**标量域内的函数**，记为$f: \mathcal{f} \to \mathbb{R}$，第二类是**切向量域内的函数**，记作$F:\mathcal{X} \to T\mathcal{X}$，在每个点$x$与切平面向量$F(x)$间建立映射关系。函数在流形上定义了标量的希尔伯特空间以及向量空间，记作$L^2(\mathcal{X})$以及$L^2(T\mathcal{X})$，内积定义为：

$$
\langle f, g \rangle_{L^2(\mathcal{X})} = \int_{\mathcal{X}}f(x)g(x)dx \\
\langle F,G \rangle_{L^2(T\mathcal{X})} = \int_{\mathcal{X}} \langle F(x),G(x) \rangle_{T_x\mathcal{X}}dx
$$

对于流形来说，没有办法直接利用公式$f(x+dx)$，为解决这一问题，定义函数$f$的微分$df: T\mathcal{X} \to \mathbb{R}$作用在切平面向量域中。对于每个点$x$，微分可被定义为作用在切平面向量域$F(x) \in T_x\mathcal{X}$上的线性函数$df(x) = \langle \nabla f(x), \cdot \rangle_{T_x\mathcal{X}}$。