# Diffusion Models: A Comprehensive Survey of Methods and Applications
### Main foundations of diffusion models
1.DDPMS[97,188,247]
2.SMS[252,253]
3.Score SDES[251,257]

## DDPMs
[97.247] two Markov chains: 1 is perturbs data to noise(hand-designed), 2 is noise back to data(deep neural networks)

重参数化 为使采样结果可导 使结果具有连接性

$q(X_t|X_{t-1} = \mathcal N(X_t;\sqrt{1-\beta_t}X_{t-1},\beta_tI)$  逐步加噪



去噪过程
$\log\mathcal p_\theta(\mathcal x_0) = \log\int{\mathcal p_\theta(\mathcal x_0,\mathcal x_1,...,\mathcal x_T)d\mathcal x_1d\mathcal x_2...d\mathcal x_T}$ 求p(x0)即使求x0的边缘概率（边缘分布），对x1...xT元素进行积分的目的是剔除他们的影响。
这个公式表示的是条件概率密度函数 $\log\mathcal p_\theta(\mathcal x_0)$ 与一个积分的对数之间的关系。让我们仔细看看这个等式的每一部分。

左边的部分是以 $\theta$ 为参数的概率密度函数 $\mathcal p_\theta(\mathcal x_0)$ 的对数。这里，$\mathcal x_0$ 代表一个变量，而 $\theta$ 代表概率密度函数的参数。通常，$\log\mathcal p_\theta(\mathcal x_0)$ 表示我们对于变量 $\mathcal x_0$ 在给定参数 $\theta$ 下的不确定性的度量。

等式右边的部分表示一个多重积分。这个积分考虑了所有可能的 $\mathcal x_1, \mathcal x_2, ..., \mathcal x_T$ 的组合，其中 $\mathcal x_1, \mathcal x_2, ..., \mathcal x_T$ 是一系列关于 $\mathcal x_0$ 的变量。这个积分的目的是计算在给定参数 $\theta$ 下，所有这些变量组合的联合概率密度函数 $\mathcal p_\theta(\mathcal x_0, \mathcal x_1, ..., \mathcal x_T)$ 的总和。

将这两部分结合在一起，我们可以解释为什么这个等式成立。这个等式实际上是在表示一个边缘化（marginalization）的过程。我们首先有一个关于多个变量（$\mathcal x_0, \mathcal x_1, ..., \mathcal x_T$）的联合概率密度函数。然后，我们通过对所有其他变量（$\mathcal x_1, \mathcal x_2, ..., \mathcal x_T$）求积分，从而得到只依赖于 $\mathcal x_0$ 的边缘概率密度函数 $\mathcal p_\theta(\mathcal x_0)$。

简而言之，等式表示了在给定参数 $\theta$ 下，一个多变量联合概率密度函数可以通过对其他所有变量求积分，得到只包含 $\mathcal x_0$ 的边缘概率密度函数

*边缘概率的作用：显示目标元素在联合概率中的影响力*