# Diffusion Models: A Comprehensive Survey of Methods and Applications
### Main foundations of diffusion models
1.DDPMS[97,188,247]
2.SMS[252,253]
3.Score SDES[251,257]

## DDPMs
[97.247] two Markov chains: 1 is perturbs data to noise(hand-designed), 2 is noise back to data(deep neural networks)

重参数化 为使采样结果可导 使结果具有连接性

$q(X_t|X_{t-1} = \mathcal N(X_t;\sqrt{1-\beta_t}X_{t-1},\beta_tI)$  逐步加噪




