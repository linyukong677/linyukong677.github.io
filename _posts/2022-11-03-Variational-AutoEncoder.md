---
layout: post
title: Ensemble
date: 2022-11-06 17:13 +0800
last_modified_at: 2022-11-07 19:08 +0800
tags: [Machine Learning]
toc:  true
math: true
---

凡富于创造性的人必敏于模仿，凡不善于模仿的人决不能创造
{: .message }

> 仔细想想，我开始接触VAE并使用VAE来解决问题是一段时间之前的事情了。VAE是一个理念很简单的模型，对于初学者而言，形象化理解很容易，但是真正要使用VAE来解决一个问题并不简单。我希望在这篇文章中，提供一个VAE解决问题的例子，并且讨论一些VAE在实际应用中的一些trick。最后我会给出VAE的数学推导。

## 一个VAE例子

> 如果没有gpu服务器，可以使用colab来运行代码。

我们以一个基础的MNIST数据集为例，来讲解VAE的使用。

> By the way，MNIST数据集是一个手写数字的数据集，每个数字都是28x28的灰度图像，每个像素点的值在0-255之间。这个数据集是机器学习中最常用的数据集之一，其地位相当于初学者学习编程时候的Hello World。对于一个新的模型，理解架构之后，在MNIST数据集上跑一遍是一个很好的验证理解的方法。

代码已上传至colab，链接为：[MINIST_VAE](https://drive.google.com/file/d/14PLEajkXxPm30m1SYXpe3dcVRxsQmbsb/view?usp=sharing)。需要注意的是，首先需要调整运行时模式，选择GPU加速。

关于这个例子，模型架构和原理部分没有多少需要解释。

## VAE Tricks



