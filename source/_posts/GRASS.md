---
title: SIGGRAPH2017 GRASS
date: 2019-10-24 10:13:47
tags:
    - Shape Analysis
    - voxel grids
    - SIGGRAPH
category: 论文阅读笔记
---

## 参考论文

GRASS: Generative Recursive Autoencoders for Shape Sturctures

## Introduction

三维模型可以通过模型中各个部分的层次结构有效地表示出来，这种层次结构能够反映出模型内部的关系，例如连接关系以及对称关系。本文希望通过一个生成网络，去表示一类物体的模型结构（shape structure）。

<!--more-->

对于三维模型的生成，应该同时考虑模型的结构以及几何细节，但是模型的结构是离散的，几何细节是连续的，因此需要设计一种网络将这两个部分结合起来。除此之外，对于一个类别的模型，例如椅子，它们之间的结构也是不相同的，如果将模型的结构用图进行表示，则需要网络能够处理不同结构以及规模的图数据。

对于一个模型的结构，包含对称以及连接的信息，这种层次关系通过图来表示，不论图的结构变化如何，都可以通过一种算法递归地将图的节点汇聚到其父节点并最终汇聚到根节点，通过这种方法，shape structure就可以通过一个固定长度的向量表示出来。论文中学习了一种神经网络，能够将模型的层次结构通过encoder编码成一个根节点编码并通过decoder解码还原模型的层次结构。之后，即可通过进一步学习根节点编码的分布生成新的根节点编码，进而生成新的模型结构。

## GRASS

GRASS网络的结构如下图所示：

![The overview of GRASS](/img/GRASS.jpg)

三维模型中的每一个部分都通过OBBs(oriented bounding boxes)表示，每一个OBB通过一个固定长度的编码表示它本身的几何细节。汇集后的编码即包含它的子OOBs的几何特征，也包含它们合并的机制（assemble by connectivity or group by symmetry）换句话说，最终生成的code具有整个模型的结构特征以及每个bounding box对应的几何特征。

### Stage 1: Recursive autoencoder

在这一部分中，自编码器将含有不同数量采用不同组织形式的OBBs映射到一个固定长度的root code中，这一过程通过递归神经网络（RvNN）实现，之后，root code又被解码恢复成原来的boxes。

### Stage 2: Learning manifold of plausible structures

在第二部分中，作者扩展了自编码器使其能够生成生成新的模型。作者采用GAN的结构，将之前自编码器的decoder作为Generator，将之前自编码器的encoder作为Discriminator，通过这一结构去学习root code的分布来描述不同的structure。

### Stage 3: Part geometry synthesis

在第三部分中，生成的通过boxes表示结构的模型需要进一步表示出详细的几何特征，给定一个合成的layout中的box，计算出在上下文当中代表它的structure-aware recursive features。之后，同时学习一种体素网格的表示以及从上述特征features到这一种表示的映射。

