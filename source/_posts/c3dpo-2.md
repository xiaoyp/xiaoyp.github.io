---
title: ICCV2019 C3DPO（二）
date: 2019-11-04 17:41:15
tags:
    - 3D reconstruction
category: ICCV
---

## Overview

C3DPO的整体结构如下图所示：

![The overview of C3DPO](/img/C3DPO.jpg)

<!--more-->

## Method

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