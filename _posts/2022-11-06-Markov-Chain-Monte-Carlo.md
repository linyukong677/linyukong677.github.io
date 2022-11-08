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

## 逻辑

我在之前的一篇文章（TODO:）里面我介绍过Markov Chain在分布采样中的使用，我认为读者仔细阅读一下之前的这篇文章会对你理解MCMC过程有很多帮助。



