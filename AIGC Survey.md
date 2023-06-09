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
$\int\mathcal p_{\theta(x,y)}dxdy = 1$
这个公式表示的是条件概率密度函数 $\log\mathcal p_\theta(\mathcal x_0)$ 与一个积分的对数之间的关系。让我们仔细看看这个等式的每一部分。

左边的部分是以 $\theta$ 为参数的概率密度函数 $\mathcal p_\theta(\mathcal x_0)$ 的对数。这里，$\mathcal x_0$ 代表一个变量，而 $\theta$ 代表概率密度函数的参数。通常，$\log\mathcal p_\theta(\mathcal x_0)$ 表示我们对于变量 $\mathcal x_0$ 在给定参数 $\theta$ 下的不确定性的度量。

等式右边的部分表示一个多重积分。这个积分考虑了所有可能的 $\mathcal x_1, \mathcal x_2, ..., \mathcal x_T$ 的组合，其中 $\mathcal x_1, \mathcal x_2, ..., \mathcal x_T$ 是一系列关于 $\mathcal x_0$ 的变量。这个积分的目的是计算在给定参数 $\theta$ 下，所有这些变量组合的联合概率密度函数 $\mathcal p_\theta(\mathcal x_0, \mathcal x_1, ..., \mathcal x_T)$ 的总和。

将这两部分结合在一起，我们可以解释为什么这个等式成立。这个等式实际上是在表示一个边缘化（marginalization）的过程。我们首先有一个关于多个变量（$\mathcal x_0, \mathcal x_1, ..., \mathcal x_T$）的联合概率密度函数。然后，我们通过对所有其他变量（$\mathcal x_1, \mathcal x_2, ..., \mathcal x_T$）求积分，从而得到只依赖于 $\mathcal x_0$ 的边缘概率密度函数 $\mathcal p_\theta(\mathcal x_0)$。

简而言之，等式表示了在给定参数 $\theta$ 下，一个多变量联合概率密度函数可以通过对其他所有变量求积分，得到只包含 $\mathcal x_0$ 的边缘概率密度函数

*边缘概率的作用：显示目标元素在联合概率中的影响力*

##### 对边缘概率的理解
假设有两个连续随机变量 X 和 Y，它们的联合概率密度函数为 p(x, y)。现在我们想要计算边缘概率密度函数 p(x)，即仅与变量 X 相关的概率分布。

为了得到 p(x)，我们需要从联合概率密度函数 p(x, y) 中消除变量 Y 的影响。我们可以通过对 Y 的所有可能取值求积分来实现这一点。换句话说，我们需要计算：

p(x) = ∫ p(x, y) dy

这里，我们在 Y 的整个取值范围内对 p(x, y) 求积分。这样，我们将 Y 的所有可能取值的概率贡献累加在一起。这个积分过程实际上计算了 X 取某个特定值时，Y 取所有可能值的情况下的总概率。因此，我们得到了只关心 X 取值的边缘概率密度函数 p(x)。

$E_{q(x_{1:T}|x_0)}[\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}]$
$E_{q(x_{1:T}|x_0)}[\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}] = ∑ [\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} * q(x_{1:T}|x_0)]$
$E_{q(x_{1:T}|x_0)}[\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}] = ∫ [\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} * q(x_{1:T}|x_0)] dx_{1:T}$


− E𝑞(x0,x1,··· ,x𝑇 ) [log 𝑝𝜃 (x0, x1, · · · , x𝑇 )]

$log p_\theta(x_0,x_1,...,x_T) = log p(x_T)+\sum_{t=1}^{T}log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}$

$KL(q(x_0,x_1,...x_T)||p\theta(x_0,x_1,...x_T))$
$E_{q(x_0,x_1,...,x_T)}[-log p(x_T)-\sum_{t=1}^{T}log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}]$

加噪过程
***整个加噪过程是一个后验估计，被表示为  $q(x_{1:T}|x_0)=\prod_{t=1}^{T}q(x_t|x_{t-1})$

加噪模型：$x_t = x_{t-1}*\alpha+\beta I$  人为定义的
$\alpha$是衰减系数，$\beta$是噪声系数，均为（0，1）之间

1.$q(x_t|x_{t-1})=\mathcal N(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI)$ 人为定义的
2.$q(x_t|x_0)=\mathcal N(x_t;\sqrt{\alpha_t}x_0,(1-\alpha_t)I)$
由于概率遵守马尔可夫链，所以在知道初始状态x0的情况下，可以直接推出xT的分布，且已知了$q(x_t|x_{t-1})$的情况下即知道了每步加噪过程，从x0进行t次叠加就可得出$q(x_t|x_0)$

关于DDPM中KL散度与损失函数的作用
KL散度作为理论分析，对为什么能生成高质量样本进行了解释。
损失函数作用于预测值与真实值实际的差距，优化模型参数以生成高质量的结果，损失函数选择的均方差与交叉熵等由实际情况而定。

$E_{t \sim \mathcal U[[1,T]],x_0 \sim q(x_0), \epsilon \sim \mathcal N(0,I)}[\lambda(t)||\epsilon_\theta(x_t,t)||^2]$



### Diffusion 总结
##### Diffusion 阶段
$X_t \~ N(\squr{\alpha})$

## Diffusion 快速采样 Efficient Sampling
1. Learning-Free Sampling 无学习抽样
2. Learning-Based Sampling 基于学习的抽样
### Learning-Free Sampling
减少时间不长，最小化离散化误差



image-to-image + 图像评分 + 路人情绪 + 开发商成本控制 + 新奇度打分