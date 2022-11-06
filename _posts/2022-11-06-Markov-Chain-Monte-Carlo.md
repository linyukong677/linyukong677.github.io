---
layout: post
title: Markov Chain Monte Carlo
date: 2022-11-06 17:13 +0800
last_modified_at: 2022-11-07 19:08 +0800
tags: [Machine Learning, Bayesian]
toc:  true
math: true
---

你附庸的附庸不是你的附庸
{: .message }

## 问题提出

在Bayesian的框架下，我们需要计算后验概率分布，即$P(\theta\mid D_{train})$，其中，$D_{train}$表示训练数据集，$\theta$是模型的参数。实际应用中，$\theta$往往是高维的，这个求后验分布的过程的难度就会很大（虽然不一样，但是可以类比高维参数空间求解问题）。一些朴素的求解算法，比如网格搜索或是Monte Carlo，都会遇到计算量过大的问题。





