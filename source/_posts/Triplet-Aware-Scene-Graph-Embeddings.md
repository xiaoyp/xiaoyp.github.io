---
title: ICCV2019 Triplet-Aware Scene Graph Embeddings
date: 2019-10-28 17:02:38
tags:
    - ICCV
category: 论文阅读笔记
---

<!--https://github.com/danfeiX/scene-graph-TF-release-->

## 参考论文

Triplet-Aware Scene Graph Embeddings

[论文链接](https://arxiv.org/pdf/1909.09256.pdf)

## Introduction

场景图（Scene graphs）作为一种结构数据用于描述两两物体之间的语义关系。在Scene graph中，每个节点代表一个物体，每一条边代表两个物体之间的关系，用一个三元组<subject, predicate, object>来表示。例如：<cat, on, road>，<dog, left of, person>。对于场景图的解释如下图所示（[D.Xu, Y.Zhu, C.Choy and L.Fei-Fei. Scene graph generation by iterative message passing. In Computer Vision and Pattern Recongnition, 2017](https://arxiv.org/pdf/1701.02426.pdf)）:

<!--more-->

![Scene Graph](/img/SceneGraph.jpg)

