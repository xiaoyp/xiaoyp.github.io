---
title: ICCV2019 C3DPO
date: 2019-11-03 20:16:20
tags:
    - 3D reconstruction
    - ICCV
category: 论文阅读笔记
---

## 参考论文

C3DPO: Canonical 3D Pose Networks for Non-Rigid Sturcture From Motion

[论文链接](https://arxiv.org/pdf/1909.02533.pdf)

## Introduction

C3DPO用于从2D的关键点的标注中提取并建立对于可变形物体的三维模型。与其他同类方法不同的是，C3DPO的训练不需要三维的信息。

C3DPO通过神经网络从二维图像中的关键点分解得到模型的视角、变形等信息，的主要贡献在于：（1）在重建三维标准模型以及视角的过程中仅使用了单一图片中的二维关键点。（2）使用一种新的自监督的限制将三维的形状以及二维的视角分离。（3）可以处理在观察过程中被遮挡的部分。（4）在多种不同类的物体上都有很好的表现。

<!--more-->

## Overview

C3DPO的整体结构如下图所示：

![The overview of C3DPO](/img/C3DPO.jpg)

## Method

### Sturcture from motion(SFM)

SFM的输入为元组$y_n = (y_{n1}, \dots, y_{nK}) \in \mathbb{R}^{2 \times K}$表示二维的关键点，其中$y_n$表示一个刚性物体（rigid object）的第n个视角。这N个视角可以被看作是通过一组三维点集、也被称为结构$X = (X_1, \dots, X_K) \in \mathbb{R}^{3 \times K}$，以及N个刚性运动$(R_n, T_n) \in SO(3) \times T(3)$生成的。因此，视角、结构、以及运动之间的关系可以被表示为$y_{nk} = \Pi(R_nX_k + Tn)$，其中$\Pi : \mathbb{R}^3 - \mathbb{R}^2$为摄像机投影函数。在这里设投影函数为线性函数，并通过一个矩阵来表示：$\Pi = [I_2, 0]$。如果所有的关键点未发生遮挡，则二维关键点与三维结构可以被中心化处理，以此忽略公式中的平移项，因此公式修改为：$y_{nk} = M_nX_k = \Pi R_n X_k$。用矩阵来表达为：

$$
Y = 
\begin{bmatrix}
y_{11} & \cdots & y_{1K} \\
\vdots & \ddots & \vdots \\
y_{N1} & \cdots & y_{NK}
\end{bmatrix},
M = 
\begin{bmatrix}
M_1 \\
\vdots \\
M_N
\end{bmatrix},
Y = MX.
$$

因此，从上述式子中可以看出，SFM可以被公式表达为将物体的视图Y分解为观察点M（viewpoint）以及结构X（structure）。但是，对于矩阵Y的分解，并不是唯一的，需要添加一些限制条件以保证矩阵分解的歧义最小。由等式$MX = (MA^{-1})(AX)$可知，只要(M, X)是解，则对任意的3*3的可逆矩阵A来说，$(MA^{-1}, AX)$都是解，可以理解为，对于这一矩阵分解的歧义有9个自由度。为保证最终矩阵分解的歧义仅来自这一可逆矩阵，需满足$2NK \ge 6N + 3K - 9$。除此之外，需要满足结构X是满秩的，即不能使得结构中存在三个关键点在同一条直线上。

### Non-rigid sturcture from motion

NR-SFM与SFM相似，但是NR-SFM能够允许物体的结构$X_n$能够在不同的view之间变形。考虑变形的约束是一个线性模型$X_n = X(\alpha_n ; S)$，其中，参数为$\alpha_n \in \mathbb{R}^D$，与视图无关的shape basis为$S \in \mathbb{R}^{3D \times K}$：

$$
X(\alpha_n; S) = (\alpha_n \otimes I_3)S
$$

其中$\alpha$是行向量，$\otimes$是克罗内克积。同理，将structure写成矩阵形式为$X = (\alpha \otimes I_3)S \in \mathbb{R}^{3N \times K}, \alpha \in \mathbb{R}^{N \times D}$。

给定关键点的多个视角，NR-SFM的目标就是从这些视角中恢复出物体的动作以及基本的形状。这一等式可被表示为:

$$
y_{nk} = \Pi(R_n\sum_{d=1}^D\alpha_{nd}S_{dk} + T_n)
$$

同样的，对其进行中心化，得到公式如下所示：

$$
Y = \bar{M}(\alpha \otimes I_3)S, (Y \in \mathbb{R}^{2N \times K}, \bar{M} \in \mathbb{R}^{2N \times 3N})
$$

矩阵$\bar{M}$为分块对角矩阵，对角矩阵上的元素为每一个view对应的变换矩阵，$\bar{M} = diag(M_1, \dots, M_N)$。

### Monocular motion and structure estimation

根据上述等式，一旦shape basis S被学习到，就可以通过物体单一的视图Y去重建viewpoint以及pose。但是，这一想法仍需要解决矩阵分解的问题。

在C3DPO中，希望训练一个映射关系：

$$
\Phi : \mathbb{R}^{2K} \times \{0, 1\}^{K} \to \mathbb{D} \times \mathbb{R}^3, (Y, v) \to (\alpha, \theta)
$$

其中，v是由布尔值组成的行向量，表示在给定的视图中，对应的关键点是否能够被看到。这一映射的输出为D维的姿态参数$\alpha$以及$\theta \in \mathbb{R}^3$，参数$\theta$用于生成相机的参数矩阵$M(\theta) = \Pi R(\theta) = \Pi expm[\theta]_{\times}$。

与其他方法相比，通过对这一映射的学习，可以包含在线性模型中不明显的物体结构信息。对于这一映射的学习，通过最小化re-projection loss来实现：

$$
\mathcal{l}_1(Y, v; \Phi, S) = \frac{1}{K} \sum_{k=1}^{K}v_k \cdot ||Y_k - M(\theta)(\alpha \otimes I_3)S_{:, k}||_{\epsilon}
$$

其中，$(\alpha, \theta) = \Phi(Y, v),||z||_{\epsilon} = (\sqrt{1+ (||z||/\epsilon)^2}-1){\epsilon}$

### Consistent factorization via canonicalization

对于NR-SFM来说，最大的挑战在于分解三维模型的视点的变化以及模型的变形带来的歧义。在这一部分中，作者提出了一种新的方法，直接去鼓励网络生成一致的模型。通常，令$\mathcal{X}_0$作为通过网络获得的所有重建模型$X(\alpha; S)$的集合。如果网络能够将viewpoints与pose分解开，那么将不存在两种重建模型$X, X' \in \mathcal{X}_0$满足$X' = RX$。也就是说，对于重建集合中的任意结构X，集合中除了其自身外不存在能够通过旋转操作得到X的结构。因此，论文中构建了标准化的函数$\Psi :\mathbb{R}^{3 \times K} \to \mathbb{R}^{3 \times K}$满足$\forall R \in SO(3), X \in \mathcal{X}_0, X = \Psi(RX)$。

在C3DPO中，为保证能够将视点与物体的姿态有效地区分开，引入loss function为：

$$
\mathcal{l}_2(X, R; \Psi) = \frac{1}{K}\sum_{k=1}^{K}||X_{:,k} - \Psi(RX)_{:,k}||_{\epsilon}
$$

其中，$R \in SO(3)$是随机地一个旋转矩阵。

在网络训练过程中，输入为$Y_n$，首先通过网络$\Phi(Y_n, v)$生成视点的参数$\theta_n$以及姿态的参数$\alpha_n$，在这一过程中，引入re-projection loss。除此之外，一个随机的旋转矩阵$\hat{R}$作用在生成的结构$X_n = X(\alpha_n; S)$上，之后，旋转后的结构$\hat{R}X_n$被送入标准化神经网络$\Psi$中，这一神经网络通过预测模型的系数$\hat{\alpha}_n$达到抵消旋转矩阵$\hat{R}$的作用,生成预测的标准模型$\hat{X}_n = X(\hat{\alpha}_n; S)$，标准模型应该能够生成未经旋转的模型$X_n$，在这一过程中，引入损失函数$l_2$。这两个网络$\Phi$与$\Psi$同时进行训练，最小化$l_1+l_2$。通过最小化上述损失函数，以达到有效分解视点与姿态的目的。

### In-plane rotation invariance

根据旋转不变性，能够进一步为神经网络添加约束用于学习。令$Y = \Pi RX$为一个三维结构X的视图。将结构X按一个轴进行旋转，即将$r_z \in SO(2)$作用在关键点上作为一个新的input，则这两个input输出的结果$\Phi(Y, v) = (\alpha, \theta)$以及$\Phi(r_zY, v) = (\alpha' , \theta')$，应该满足$\alpha = alpha'$。因此，修改reprojection loss function为：

$$
l_3(Y, v; \Phi, S) = \frac{1}{K}\sum_{k=1}^{K}v_{k} ||r_zY_k - M(\theta')(\alpha \otimes I_3)S_{:,k}||_{\epsilon}
$$

因此，最终的loss修改为$l_2+l_3$。

（论文里提到$\Phi(Y, v) = (\alpha, \theta)$以及$\Phi(r_zY, v) = (\alpha' , \theta')$，旋转后的view与原始的view用了一个v,但是旋转后关键点的遮挡情况可能会不一样，因此，我觉得不应该用一个v，不清楚这一点会对结果有什么影响）
