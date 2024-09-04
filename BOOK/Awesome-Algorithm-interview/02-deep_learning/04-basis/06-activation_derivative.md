> 常用的激活函数，以及对应的导数？

在神经网络中，激活函数起着将线性输入转换为非线性输出的重要作用，从而使神经网络能够学习和表示复杂的模式。以下是一些常用的激活函数及其对应的导数：

### 1. Sigmoid 函数

**定义**：
$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

**导数**：
$$ \sigma'(x) = \sigma(x) (1 - \sigma(x)) $$

**特点**：
- **范围**：输出值在 (0, 1) 之间。
- **优点**：平滑且连续，广泛应用于二分类问题的输出层。
- **缺点**：在极端值附近梯度接近于0，导致梯度消失问题。

### 2. Tanh 函数

**定义**：
$$ \text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

**导数**：
$$ \text{tanh}'(x) = 1 - \text{tanh}^2(x) $$

**特点**：
- **范围**：输出值在 (-1, 1) 之间。
- **优点**：比Sigmoid函数中心化（输出均值为0），梯度更大，训练更快。
- **缺点**：同样存在梯度消失问题。

### 3. ReLU (Rectified Linear Unit) 函数

**定义**：
$$ \text{ReLU}(x) = \max(0, x) $$

**导数**：
$$ \text{ReLU}'(x) = \begin{cases} 
1 & \text{if } x > 0 \\
0 & \text{if } x \leq 0 
\end{cases} $$

**特点**：
- **范围**：输出值在 [0, ∞) 之间。
- **优点**：简单高效，计算速度快，能有效缓解梯度消失问题。
- **缺点**：在负区间梯度为0，导致神经元可能会“死掉”。

### 4. Leaky ReLU 函数

**定义**：
$$ \text{Leaky ReLU}(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0 
\end{cases} $$
其中，$\alpha$ 是一个小的常数，一般取值如 0.01。

**导数**：
$$ \text{Leaky ReLU}'(x) = \begin{cases} 
1 & \text{if } x > 0 \\
\alpha & \text{if } x \leq 0 
\end{cases} $$

**特点**：
- **范围**：输出值在 (-∞, ∞) 之间。
- **优点**：解决了ReLU的“死亡”问题，允许负值通过。
- **缺点**：相比ReLU，计算稍复杂。

### 5. ELU (Exponential Linear Unit) 函数

**定义**：
$$ \text{ELU}(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha (e^x - 1) & \text{if } x \leq 0 
\end{cases} $$
其中，$\alpha$ 是一个大于0的常数。

**导数**：
$$ \text{ELU}'(x) = \begin{cases} 
1 & \text{if } x > 0 \\
\alpha e^x & \text{if } x \leq 0 
\end{cases} $$

**特点**：
- **范围**：输出值在 (-α, ∞) 之间。
- **优点**：结合了ReLU和Leaky ReLU的优点，在负区间有非零梯度，输出值均值接近于零。
- **缺点**：计算复杂度比ReLU高。

### 6. Swish 函数

**定义**：
$$ \text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} $$

**导数**：
$$ \text{Swish}'(x) = \text{Swish}(x) + \sigma(x) (1 - \text{Swish}(x)) $$

**特点**：
- **范围**：输出值在 (-∞, ∞) 之间。
- **优点**：平滑且无界，能自适应调整输出，表现优于ReLU。
- **缺点**：计算复杂度较高。

### 参考资料

1. [Activation Functions](https://www.deeplearningbook.org/): "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
2. [Understanding the Gradient of Activation Functions](https://towardsdatascience.com/activation-functions-and-its-derivatives-6d9c94fd2d53)
3. [Activation Functions in Neural Networks](https://www.analyticsvidhya.com/blog/2021/05/complete-guide-on-activation-functions/)