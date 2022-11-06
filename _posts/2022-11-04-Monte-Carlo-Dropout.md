---
layout: post
title: Monte Carlo Dropout
date: 2022-11-04 19:13 +0800
last_modified_at: 2022-11-04 20:08 +0800
tags: [Bayesian, Machine Learning]
toc:  true
math: true
---

对于 Dropout 的小小修改可以使得一个一般的神经网络成为一个 Bayesian 神经网络。
{: .message }

## Preliminaries

Dropout 是一种在训练神经网络时使用的正则化方法。在训练时，每次迭代时，随机将一部分神经元的权重置为 0。这样做的目的是为了防止过拟合。在测试时，不使用 Dropout。

贝叶斯学派的一个特点就是，不同于频率学派得到的结果是一个确定的模型，贝叶斯学派得到的结果是一个关于模型的分布。

## MC Dropout

我们将Preliminaries中两点结合起来，得到一个新的方法：MC Dropout。在训练时，每次迭代时，随机将一部分神经元的权重置为 0。在测试时，前向传播也要使用Dropout进行Prediction，但是进行多次Prediction。最后，将多次Prediction的结果取平均值作为最终的结果，而多次Prediction的方差可以作为不确定性的度量。

### Gaussian Dropout

Gaussian Dropout 是一种对 Dropout 的改进。传统的 Dropout 是将神经元的权重置为 0，而 Gaussian Dropout 是将神经元的权重置为一个随机的高斯分布。这种 Gaussian Dropout 的方式同样可以用来做 MC Dropout。

## Discussion

我个人认为，MC Dropout 是一次将Deep Learning模型与贝叶斯学派结合起来的尝试。但是，MC Dropout 有一些自身目前无法回避的问题：

1. dropout 本身有一些随机性，因而带来的 Uncertainty 只是具有参考价值而不是一个较为精准的度量。
2. MC Dropout 本身的计算量较大。为了尽量减小第一点中提到的随机性，实际中使用MC Dropout的时候，需要进行很多次的前向传播。这样的计算量是较大的。

## Code

我实现了一个简单的 MC Dropout 的例子，可以在这里找到（TODO:）。

