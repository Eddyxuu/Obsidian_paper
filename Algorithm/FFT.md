
多项式相乘
- Brute-Forced 各个子项相乘 $O(N^2)$
- FFT（fast Fourier transform）$O(Nlog_2{N})$

## 步骤

将多项式分为奇偶部分
相反数的值相等
让新的求值点也分为相反数对  -- 即用到复数   （树形结构 $i^2 = -1$  , 平方后仍未正反数
ps. 因为每个求值点的平方都是正的，所以无法完成递归操作，才引入复数概念


coeff->value  == evaluation
![[Pasted image 20230522192619.png]]