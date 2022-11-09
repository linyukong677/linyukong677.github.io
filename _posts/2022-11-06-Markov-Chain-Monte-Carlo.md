---
layout: post
title: Markov Chain Monte Carlo
date: 2022-11-06 17:13 +0800
last_modified_at: 2022-11-07 19:08 +0800
tags: [Machine Learning, Bayesian]
toc:  true
math: true
---

MCMC算法的简单逻辑分析
{: .message }

## 问题提出

在Bayesian的框架下，我们需要计算后验概率分布，即$$P(\theta\mid D_{train})$$，其中，$$D_{train}$$表示训练数据集，$$\theta$$是模型的参数。实际应用中，$$\theta$$往往是高维的，这个求后验分布的过程的难度就会很大（虽然不一样，但是可以类比高维参数空间求解问题）。一些朴素的求解算法，比如网格搜索或是Monte Carlo，都会遇到计算量过大的问题。

## 逻辑

我在之前的一篇文章（TODO:）里面我介绍过Markov Chain在分布采样中的使用；在另一篇文章（TODO:）中我介绍过Monte Carlo方法，我认为读者必须仔细阅读一下之前的这两篇文章，否则接下来的逻辑无从理解。

OK，现在我们来看MCMC的逻辑。MCMC解决的问题是要去解后验分布$$P(\theta\mid D_{train})$$。首先，Markov Chain可以得到对平稳分布的采样。如果我们构造一种Markov Chain，使得它的平稳分布就是我们要求的后验分布$$P(\theta\mid D_{train})$$，那么我们就可以通过这个Markov Chain来得到后验分布的采样。但是采样不等于分布，这就回到了Monte Carlo方法的思想上，我们可以通过采样的结果来估计后验分布有关的数学计算量（比如数学期望$$E_{P(\theta\mid D_{train})}[\theta]$$）。而在Bayesian一节中我们知道，后验分布的数学期望就是我们用来做预测的参数$$\theta$$，所以，借助MCMC方法，我们虽然无法得到后验分布$$P(\theta\mid D_{train})$$的数学表达式，但是我们可以得到后验分布的采样，从而计算出我们需要的值。

