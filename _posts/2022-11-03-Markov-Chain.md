---
layout: post
title: Markov Chain
date: 2022-11-06 17:13 +0800
last_modified_at: 2022-11-07 19:08 +0800
tags: [Machine Learning, Bayesian]
toc:  true
math: true
---

你附庸的附庸不是你的附庸
{: .message }

## Markov Chain Introduction

Markov Chain的定义本身很简单：在一个状态转移过程中，某一时刻状态转移的概率只依赖于它的前一个状态。举个不太恰当的例子，假如每天的天气是一个状态的话，那个今天是不是晴天只依赖于昨天的天气，而和前天的天气没有任何关系，当然现实中并不是这样，但是这不妨碍我们将天气变化问题抽象地简化为一个Markov Chain。Markov Chain的数学定义为：

$$
P(x_{t}\mid x_{t-1},\ldots,x_{1})=P(x_{t}\mid x_{t-1})
$$

得益于Markov Chain对于状态转移的简化，Markov Chain在序列模型中有着广泛应用，如循环神经网络RNN，隐式马尔科夫模型HMM等。

## Markov 状态转移矩阵

由于Markov Chain的状态转移概率只依赖于前一个状态，因此我们可以将Markov Chain的状态转移的概率分布用一个矩阵来表示，这个矩阵就是Markov状态转移矩阵。比如如下这个例子：

![Marikov Chain Model.png](https://s2.loli.net/2022/11/06/SOGlfPbI7VdzCk9.png)

这个马尔科夫链是表示股市模型的，共有三种状态：牛市（Bull market）, 熊市（Bear market）和横盘（Stagnant market）。每一个状态都以一定的概率转化到下一个状态。比如，牛市以0.025的概率转化到横盘的状态。这个状态概率转化图可以以矩阵的形式表示。如果我们定义矩阵阵P某一位置$$P(i,j)$$的值为$$P(j\mid i)$$,即从状态$$i$$转化到状态$$j$$的概率，并定义牛市为状态0， 熊市为状态1, 横盘为状态2. 这样我们得到了马尔科夫链模型的状态转移矩阵为：

$$
\begin{pmatrix}
	0.9 & 0.075 & 0.025 \\
	0.15 & 0.8 & 0.05 \\
	0.25 & 0.25 & 0.5 \\
\end{pmatrix}
$$

我们来尝试这样一个实验：初始时我们设置牛市（Bull market）, 熊市（Bear market）和横盘（Stagnant market）的概率为$$[0.3,0.4,0.3]$$，我们用上面的这个Markov Chain迭代100次，看一下概率分布的变化情况，代码如下：

```python
import numpy as np

mc_matrix=np.array([[0.9,0.075,0.025],[0.15,0.8,0.05],[0.25,0.25,0.5]], dtype=np.float32)
vector1 = np.matrix([[0.3,0.4,0.3]], dtype=np.float32) ## the initial distribution

for i in range(100):
  vector1=vector1*mc_matrix
  print("Current Round: ",i+1)
  print(vector1.tolist())
```

跑一下这个实验，你会发现最后概率分布收敛到了$$[0.625, 0.312, 0.062]$$这样的一个分布。这个是巧合吗？在这里欢迎读者多尝试几种初始的概率分布做一下实验，你会发现，概率分布最后都会收敛，而且都会收敛到$$[0.625, 0.312, 0.062]$$这个分布上。

OK，你应该可以猜到了，首先，Markov Chain的概率分布最后会收敛，其次，收敛到的概率分布是一个固定的、由Markov Chain自身决定的分布，与初始时的分布无关。这个性质不光对我们上面的状态转移矩阵有效，对于绝大多数的其他的马尔科夫链模型的状态转移矩阵也有效。同时不光是离散状态，连续状态时也成立。

进一步地，我们可以尝试这个实验：

```python
import numpy as np

mc_matrix=np.array([[0.9,0.075,0.025],[0.15,0.8,0.05],[0.25,0.25,0.5]], dtype=np.float32)
matrix=mc_matrix.copy()

for i in range(100):
  matrix=np.matmul(matrix,mc_matrix)
  print("Current Round: ",i+1)
  print(matrix.tolist())
```

这个实验的结果是，最后的状态转移矩阵收敛到了一个固定的矩阵，这个矩阵就是我们上面的状态转移矩阵的100次幂。而收敛到的矩阵的每一行是$$[0.625, 0.312, 0.062]$$

我们对于这个性质进行形式化描述：如果一个非周期的马尔科夫链有状态转移矩阵$$P$$, 并且它的任何两个状态是连通的，那么$$\lim\limits_{n→∞}P_{ij}^n$$与$$i$$无关，我们有：

$$
\lim\limits_{n→∞}P_{ij}^n=\pi(j)
$$

$$
\lim\limits_{n→∞}P^n=\begin{pmatrix}
	\pi(1) & \cdots & \pi(m) \\
	\vdots &  & \vdots \\
	\pi(1) & \cdots & \pi(m) \\
\end{pmatrix}
$$

$$
\pi(j)=\sum\limits_{i=1}^m\pi(i)P_{ij}
$$

这里$$\pi(j)$$表示的是状态$$j$$的概率，$$\pi(i)$$表示的是状态$$i$$的概率，$$P_{ij}$$表示的是状态$$i$$转移到状态$$j$$的概率。其中，$$\pi$$可以通过求解下面的方程得到：

$$
\pi^T P=\pi^T
$$

## Markov Chain 采样

Markov Chain的这个优秀的收敛性带来了Markov Chain在采样当中的应用。具体来说采样的流程为：

1. 输入马尔科夫链状态转移矩阵$$P$$，设定状态转移次数阈值$$n_1$$，需要的样本个数$$n_2$$
2. 从任意简单概率分布采样得到初始状态值$$x_0$$
3. ```for t=0 to n1+n2−1: ```从条件概率分布$$P(x\mid x_t)$$中采样得到样本$$x_{t+1}$$

样本集$$(x_{n_1},x_{n_1+1},...,x_{n_1+n_2−1})$$即为我们需要的平稳分布对应的样本集。
