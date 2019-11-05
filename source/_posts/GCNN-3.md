---
title: GCNN 图卷积网络(三) 
date: 2019-10-30 21:52:42
tags: Graph
category: Deep Learning
---

## 相关论文1： Convolutional Neural Networks on Graphs with Fast Localized Spetral Filtering

### Learning Fast Localized Spectral Filters

#### Recursive formulation for fast filtering

在上文中，我们得到经过filtering操作之后的信号输出为$y = g_{\theta}(L)x = \sum_{k=0}^{K-1}\theta_kT_k(\widetilde{L})x$，其中$\widetilde{L} = 2L/\lambda_{max} - I_n$。

<!--more-->

为尽可能降低计算的复杂度，令$\bar{x}_k = T_k(\widetilde{L})x \in \mathbb{R}^n$。之后，采用递归公式$\bar{x}_k = 2\widetilde{L}\bar{x}_{k-1} - \bar{x}_{k-2}$（$\bar{x}_0=x, \bar{x}_1=\widetilde{L}x$）。通过这一递归公式，使得之前的矩阵乘法转变为矩阵与向量相乘，最终filtering的计算转变为$y = g_\theta(L)x = [\bar{x}_0, \dots, \bar{x}_{K-1}]\theta$，因此，最终计算的时间复杂度为$O(K|\mathcal{E}|)$。

### GCN代码

