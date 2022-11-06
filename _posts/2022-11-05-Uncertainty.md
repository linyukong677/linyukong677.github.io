---
layout: post
title: Uncertainty Quantification
date: 2022-11-05 19:13 +0800
last_modified_at: 2022-11-06 11:08 +0800
tags: [Baysian, Machine Learning]
toc:  true
---

一个人可贵的不是能做什么，而是知道自己能做什么
{: .message }

目前深度学习在很多领域的表现都非常好，像是CV和NLP领域都能达到很高的准确率。但是众所周知，Tesla无人驾驶之前发生了很多起事故。其中一起事故的最终原因是视觉算法误将一辆浅色卡车误判为天空。

这个事故暴露出来的问题是，传统的学习算法得到的模型只能做出预测，并且保证整体的预测具有一定的准确度，但是具体到每一次预测，模型没法给出一个量化的**Confidence**。这便是 **Uncertainty Quantification** 解决的问题。

> 举个例子，已知 Alice 15岁，Alice 和 Bob 同班，预测 Bob 几岁。这里 Alice 15岁相当于训练数据，Alice 和 Bob 同班相当于模型。在这个语境下，对于人类，我们只能说 Bob 大概率也是15岁，但是交给一个传统的深度学习模型，模型就会确定地给出 Bob 的年龄，而无法告诉你这样的预测有多不确定。

如果你了解过 Supervised Learning 中的 Classification 问题，你可能会联想到，Classification 模型最后往往会过一层 Sigmoid 或 Softmax，从而将模型的输出转换为概率值。需要区别一下，这个概率不属于 Uncertainty，描述模型对于输出这个概率有多大的把握，这个才属于我们接下来研究的 Uncertainty。

衡量 Uncertainty 的技术就是 **Uncertainty Quantification**（UQ）

## Preliminaries

在我们深入研究 UQ 之前，我们需要先了解一些前置知识，包括：

- **Baysian Machine Learning**
- **Deep Learning**
- **Aleatoric and Epistemic Uncertainty**

其中，Deep Learning 相信读者已经有一定的了解，这里不再赘述。Baysian Machine Learning 我在（TODO:）一文中有过详细介绍。

Aleatoric Uncertainty 和 Epistemic Uncertainty 是 UQ 中的两个重要概念，这里我们先简单介绍一下。

- **Aleatoric Uncertainty**（随机不确定性）：这种不确定性来自于训练数据集本身，比如，存在错误数据，或者数据本身存在不确定性。比如我们训练一个根据年龄预测身高的模型，input 为年龄，output 为身高。在训练数据中，大概率会出现相同年龄的人身高不同的情况，这就导致了模型不可能学到一个确定的映射规律，因此模型的预测结果就会有一定的随机性。这种不确定性就是 Aleatoric Uncertainty。
- **Epistemic Uncertainty**（认知不确定性）：这种不确定性来自于模型本身，比如，模型的表达能力不足，或者模型的训练数据不足。假设我们训练一个分类人脸和猩猩脸的模型，训练中没有做任何的增强，也就是说没有做数据集的旋转，模糊等操作。如果我给模型一个正常的人脸，或者是正常猩猩的脸，我们的模型应该对他所产生的结果的置信度很高。但是如果我给他猫的照片，一个模糊处理过得人脸，或者旋转90°的猩猩脸，模型的置信度应该会特别低。换句话说，认知不确定性测量的，是我们的input data是否存在于**已经见过且记住的数据**的分布之中。

## Uncertainty Quantification

我个人 Uncertainty Quantification 的方法大致分为两大类：

- **Uncertainty Quantification using Baysian Techniques**
- **Uncertainty Quantification in Reinforcement Learning**
- **Uncertainty Quantification using Ensemble Techniques**

### Uncertainty Quantification using Baysian Techniques

我在（TODO:）一文中详细介绍过 Baysian Machine Learning，整体来说，Baysian 的思想是，我们不再去学习一个确定的模型，而是学习一个模型的分布。这样，我们就可以通过对模型的分布进行采样，来获得不同的模型，从而获得不同的预测结果。那么，模型的分布的多样性（Variance），就可以用来衡量 Uncertainty。

我们主要介绍这几种 Baysian 方法：

- Monte Carlo Dropout (MC Dropout)
- Markov Chain Monte Carlo (MCMC)
- Variational Inference (VI)
- Baysian Active Learning (BAL)
- Baysian By Backprop (BBP)
- Variational Autoencoder (VAE)
- Laplacian Approximation (LA)

#### Monte Carlo Dropout (MC Dropout)

对于MC Dropout，我有一篇专门的介绍和代码实例，建议阅读：（TODO:）

其实看完上面的文章，MC Dropout 的原理就很清楚了。总的来说，MC Dropout 是利用了模型训练中的 Dropout，将训练出来的确定性模型转换成了一个模型的分布，从而将Deep Learning与Baysian结合起来。而这个模型的分布，就可以用来衡量 Uncertainty。


### Uncertainty Quantification using Ensemble Techniques

## Refererence

[1]: [A review of uncertainty quantification in deep learning: Techniques, applications and challenges](https://www.sciencedirect.com/science/article/pii/S1566253521001081)
