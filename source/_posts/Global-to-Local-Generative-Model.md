---
title: SIGGRAPH18 Global-to-Local Generative Model
date: 2019-10-26 20:13:27
tags: 
    - voxel grids
category: SIGGRAPH
---

## 参考论文

Global-to-Local Generative Model for 3D Shapes

[论文链接](https://vcc.tech/file/upload_file/image/research/att201809231254/G2L.pdf)

<!--more-->

## Introduction

对抗生成网络（GAN）在非常多的领域中被广泛应用，并取得了非常好的效果，通常来说，GAN能够很好地从类的分布中取样，但是对细节的生成不够好。而对于3D Shape来说，GAN虽然能够很好地生成三维模型的结构，但是很难生成好的几何细节。

在这篇论文中，作者提出了一种Global to Local(G2L)的生成模型用于生成3D Shape，这一模型首先通过对抗生成网络去生成全局的结构以及局部的标签，之后，全局的判别器用于区分真实与生成的3D Shape，而局部的判别器用于区分局部的每一部分，最后，再通过条件自编码器去提高局部部分的生成质量。同时，作者通过进一步优化额外的两类loss去保证模型生成的效果，第一，尽量保证每一个部分中体素的标签尽可能统一，第二，尽量保证生成模型的表面是平滑的。

## The Overview of G2L

Global-to-Local Generative Model的结构图如下所示：

![G2L](/img/G2L.jpg)

这一生成模型由两部分组成，包括左半部分的Global-to-Local GAN(G2LGAN)以及右半部分的Part Refiner(PR)。

### G2LGAN

对于GAN网络来说，存在两个挑战需要解决，第一，需要解决“model collapse”的问题，即在优化过程中，生成器会更倾向于去生成简单的、易判断为真的模型；第二，需要解决收敛的问题。在本文中，作者利用了[WGAN-NP](https://arxiv.org/pdf/1704.00028.pdf)去解决这两个问题。判别器包含全局判别器以及局部判别器，实现对生成模型整体结构以及局部的判别，但是，通过GAN生成的模型分辨率较低，且模型数据比较稀疏，因此很难仅通过GAN就能获得很好地生成效果。因此需要继续在G2LGAN之后添加PR以保证生成更好的局部细节。

在训练GAN的过程中，主要需要解决生成的shape的体素容易不相邻以及标注的part label在两个part之间容易混到一起，作者提出了三点作为对这两种情况的解决方案：第一，local discriminator不只一个，每个的discriminator都专注于判断一个特定的part；第二，通过smoothness loss减少生成不连接的体素；第三，通过purity loss去避免相邻part之间体素的标签混合到一起。

### PR

在PR部分，所有的part被依次送入统一的自编码器中进行训练，而每一个体素所对应的part label也被同时送入网络中训练，以得到能够训练不同part的网络模型。通过PR网络的训练，原始的Shape从$32^3$扩展为$64^3$，并且去除了通过GAN生成的网络中不连接的体素。



