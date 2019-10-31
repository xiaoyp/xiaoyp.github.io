---
title: ICCV2019 PU-GAN(二)
date: 2019-10-31 20:33:41
tags:
    - upsampling
category: ICCV
---

## 参考论文

PU-GAN: a Point Cloud Upsampling Adversarial Network

[论文链接](https://arxiv.org/pdf/1907.10844.pdf) 

## Loss function

### Adversarial loss

$$
\mathcal{L}_{gan}(G) = \frac{1}{2}[D(\mathcal{Q}) - 1]^2 
$$

$$
\mathcal{L}_{gan}(D) = \frac{1}{2}[D(\mathcal{Q})^2 + (D(\hat{\mathcal{Q}}) - 1)^2]
$$

其中$D(\mathcal(Q))$表示由生成器生成的点集$\mathcal{Q}$通过判别器D之后的confidence值。

<!--more-->

### Uniform loss

为使得最后生成的点云集合在目标物体的表面上分布均匀，需要设计Uniform loss来满足这一要求。论文中提到，在网络的训练中，首先通过farthest sampling在生成的$\mathcal{Q}$中取出M个点作为种子，之后以这些种子为园心，以$r_d$为半径，将圆内的点聚合到一个M个子集$S_j , j=1 \dots M$中。这一圆按照测地距离进行划分，在进行这些操作之前，首先需要对整个模型进行归一化，令整个模型平铺之后的区域约为$\pi 1^2$，则每个子集中的点与所有点的百分比应大致为$\pi r^2_d / \pi 1^2 = r^2_d$，通过这一公式计算出一个子集中应有的点数，记为$\hat{n}$，则定义下述公式：

$$
U_{imbalance}(S_j) = \frac{(|S_j| - \hat{n})^2}{\hat{n}}
$$

但是，仅通过上述约束依然不够，在同一个子集中，仍然会有很多不同的点的分布，因此需要继续考虑在同一个子集中的点局部分布的情况。

在一个点的子集$S_j$中，我们去找到其中每一个点距离最近的邻居，对于子集中的第k个点，计算其距离为$d_{j,k}$，如果$S_j$内的点满足均匀分布，则每个点到它最近点的距离$\hat{d}$应该大致为$\sqrt{\frac{2\pi r_d^2}{|S_j|\sqrt{3}}}$，这一公式假设$S_j$所代表的子集为平面且每个点代表一个六边形，如下图所示：

![Uniform loss](/img/PUGANUniform.jpg)

因此定义下述公式：

$$
U_{clutter}(S_j) = \sum_{k=1}^{|S_j|}\frac{(d_{j,k}-\hat{d})^2}{\hat{d}}
$$

因此，得到最终的Uniform loss为：

$$
\mathcal{L}_{uni} = \sum_{j=1}^{M}U_{imbalance}(S_j)\cdot U_{clutter}(S_j)
$$

### Reconstruction loss

上述两种loss函数其实并没有限制点云是否覆盖在目标模型的表面上，因此需要继续引入reconstruction loss：

$$
\mathcal{L}_{rec} = \min_{\phi : \mathcal{Q} \to \hat{\mathcal{Q}}} \sum_{q_i \in \mathcal{Q}} ||q_i - \phi (q_i)||_2
$$

其中$\phi : \mathcal{Q} \to \hat{\mathcal{Q}}$是双射映射。

### Compound loss

总的来看，我们通过最小化$\mathcal{L}_{G}$以及$\mathcal{L}_{D}$来训练PU-GAN:

$$
\mathcal{L}_G = \lambda_{gan}\mathcal{L}_{gan}(G) + \lambda_{rec}\mathcal{L}_{rec} + \lambda_{uni}\mathcal{L}_{uni}
$$

$$
\mathcal{L}_D = \mathcal{L}_{gan}(D)
$$