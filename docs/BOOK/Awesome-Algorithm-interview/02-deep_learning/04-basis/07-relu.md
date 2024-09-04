> relu的有优点？又有什么局限性？他们的系列改进方法是啥？


### ReLU的优点与局限性及其改进方法

#### ReLU的优点

1. **简单高效**：
   - 计算简单，只需进行一次比较操作 $\max(0, x)$，计算速度快。

2. **加速收敛**：
   - 相较于Sigmoid和Tanh，ReLU能加速神经网络的训练过程。因为ReLU的导数恒为1，不会出现梯度消失的问题，从而加快梯度下降算法的收敛速度。

3. **稀疏激活**：
   - ReLU会使一部分神经元输出为零，导致网络在某些层的稀疏性，从而提高模型的计算效率和内存利用率。

#### ReLU的局限性

1. **死亡ReLU问题**：
   - 当神经元的输入总是负值时，该神经元将永远不会被激活（输出始终为零），导致神经元“死亡”。这种现象通常发生在使用较大学习率时。

2. **不平衡的梯度**：
   - ReLU的输出不平衡（输出值在[0, ∞)），可能导致梯度不平衡问题，进而影响训练稳定性。

#### ReLU的改进方法

1. **Leaky ReLU**

   **定义**：
   $$
   \text{Leaky ReLU}(x) = \begin{cases} 
   x & \text{if } x > 0 \\
   \alpha x & \text{if } x \leq 0 
   \end{cases}
   $$
   其中，$\alpha$ 是一个小常数，通常取值如0.01。

   **优点**：
   - 解决了死亡ReLU问题，在负值区域有小的非零梯度，允许一些负值通过，从而使得神经元有可能被重新激活。

2. **Parametric ReLU (PReLU)**

   **定义**：
   $$
   \text{PReLU}(x) = \begin{cases} 
   x & \text{if } x > 0 \\
   \alpha x & \text{if } x \leq 0 
   \end{cases}
   $$
   其中，$\alpha$ 是一个可学习的参数。

   **优点**：
   - 提供了更大的灵活性，因为 $\alpha$ 是通过训练数据学习得到的，能更好地适应不同的任务。

3. **Exponential Linear Unit (ELU)**

   **定义**：
   $$
   \text{ELU}(x) = \begin{cases} 
   x & \text{if } x > 0 \\
   \alpha (e^x - 1) & \text{if } x \leq 0 
   \end{cases}
   $$
   其中，$\alpha$ 是一个大于0的常数。

   **优点**：
   - 在负值区域有非零输出，使得输出均值接近于零，从而提高训练速度。
   - 能够缓解死亡ReLU问题，并且输出更平滑，具有更好的梯度特性。

4. **Scaled Exponential Linear Unit (SELU)**

   **定义**：
   $$
   \text{SELU}(x) = \lambda \begin{cases} 
   x & \text{if } x > 0 \\
   \alpha (e^x - 1) & \text{if } x \leq 0 
   \end{cases}
   $$
   其中，$\lambda$ 和 $\alpha$ 是固定的参数。

   **优点**：
   - SELU在深度网络中具有自正则化的特性，能够自动保持激活值的均值和方差稳定。

5. **Swish**

   **定义**：
   $$
   \text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
   $$

   **优点**：
   - 平滑且无界，能自适应调整输出，表现优于ReLU。

### 参考资料

1. [Rectified Linear Unit (ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
2. [Understanding Leaky ReLU](https://towardsdatascience.com/understanding-leaky-relu-and-why-you-should-use-it-a77b4c3f3c5d)
3. [Exponential Linear Unit (ELU) and Scaled ELU (SELU)](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
4. [Swish: A Self-Gated Activation Function](https://arxiv.org/abs/1710.05941)