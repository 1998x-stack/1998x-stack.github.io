> sigmoid和tanh为什么会导致梯度消失？


### Sigmoid和Tanh导致梯度消失的原因

#### 1. Sigmoid函数

**定义**：
$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

**导数**：
$$ \sigma'(x) = \sigma(x)(1 - \sigma(x)) $$

**原因**：
- **输出范围**：Sigmoid函数的输出范围在 (0, 1) 之间。当输入值较大或较小时，输出值会趋近于0或1。
- **梯度**：在这些区域，Sigmoid函数的导数会变得非常小。例如，当输入值 $x$ 很大时， $\sigma(x) \approx 1$，则 $\sigma'(x) \approx 0$。同样地，当输入值 $x$ 很小时， $\sigma(x) \approx 0$，则 $\sigma'(x) \approx 0$。
- **梯度传递**：在深度网络中，这些小的梯度值通过链式法则传递到前层时，会被多次乘小数，导致梯度在传递过程中迅速减小，甚至接近于零，从而使得前层的权重几乎不更新。

#### 2. Tanh函数

**定义**：
$$ \text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

**导数**：
$$ \text{tanh}'(x) = 1 - \text{tanh}^2(x) $$

**原因**：
- **输出范围**：Tanh函数的输出范围在 (-1, 1) 之间。当输入值较大或较小时，输出值会趋近于1或-1。
- **梯度**：在这些区域，Tanh函数的导数会变得非常小。例如，当输入值 $x$ 很大时， $\text{tanh}(x) \approx 1$，则 $\text{tanh}'(x) \approx 0$。同样地，当输入值 $x$ 很小时， $\text{tanh}(x) \approx -1$，则 $\text{tanh}'(x) \approx 0$。
- **梯度传递**：在深度网络中，这些小的梯度值通过链式法则传递到前层时，会被多次乘小数，导致梯度在传递过程中迅速减小，甚至接近于零，从而使得前层的权重几乎不更新。

### 解决方法

1. **使用ReLU激活函数**：
   - ReLU（Rectified Linear Unit）函数定义为：
     $$
     \text{ReLU}(x) = \max(0, x)
     $$
   - ReLU的导数为1或0，避免了梯度消失问题。

2. **批归一化（Batch Normalization）**：
   - 对每一层的输出进行归一化，减小内部协变量偏移，保持激活值在合理范围内。

3. **适当的权重初始化**：
   - 使用如Xavier初始化或He初始化等方法，根据网络层数和激活函数特点合理初始化权重，避免初始阶段的梯度消失问题。

### 参考资料

1. [Understanding the vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
2. [Why does the sigmoid and tanh activation function lead to vanishing gradients?](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)
3. [Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](http://www.deeplearningbook.org/)