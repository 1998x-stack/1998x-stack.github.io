> 以一层隐层的神经网络，relu激活，MSE作为损失函数推导反向传播

### 反向传播推导：一层隐层的神经网络，ReLU激活，MSE损失函数

假设我们有一个简单的神经网络，包含一个隐层，使用ReLU激活函数，并且采用均方误差（MSE）作为损失函数。推导反向传播的步骤如下：

#### 前向传播

1. **输入层到隐层**：
   - 输入：$\mathbf{x} \in \mathbb{R}^n$
   - 权重：$\mathbf{W_1} \in \mathbb{R}^{m \times n}$
   - 偏置：$\mathbf{b_1} \in \mathbb{R}^m$
   - 隐层输出（未激活）：$\mathbf{z_1} = \mathbf{W_1} \mathbf{x} + \mathbf{b_1}$

2. **隐层激活**：
   - 激活函数：ReLU
   - 激活输出：$\mathbf{a_1} = \text{ReLU}(\mathbf{z_1}) = \max(0, \mathbf{z_1})$

3. **隐层到输出层**：
   - 权重：$\mathbf{W_2} \in \mathbb{R}^{k \times m}$
   - 偏置：$\mathbf{b_2} \in \mathbb{R}^k$
   - 输出层输出：$\mathbf{z_2} = \mathbf{W_2} \mathbf{a_1} + \mathbf{b_2}$

4. **输出**：
   - 模型输出：$\hat{\mathbf{y}} = \mathbf{z_2}$

#### 损失函数

- 目标值：$\mathbf{y} \in \mathbb{R}^k$
- 损失函数：均方误差（MSE）
$$ L = \frac{1}{2} \|\mathbf{y} - \hat{\mathbf{y}}\|^2 = \frac{1}{2} \sum_{i=1}^k (y_i - \hat{y}_i)^2 $$

#### 反向传播

1. **输出层到隐层**：
   - 计算损失对输出层激活值的梯度：
   $$
   \frac{\partial L}{\partial \hat{y}_i} = \hat{y}_i - y_i
   $$
   - 由于 $\hat{\mathbf{y}} = \mathbf{z_2}$，因此：
   $$
   \frac{\partial L}{\partial z_{2i}} = \hat{y}_i - y_i
   $$
   - 计算损失对隐层输出的梯度：
   $$
   \frac{\partial L}{\partial \mathbf{W_2}} = \frac{\partial L}{\partial \mathbf{z_2}} \cdot \mathbf{a_1}^T
   $$
   $$
   \frac{\partial L}{\partial \mathbf{b_2}} = \frac{\partial L}{\partial \mathbf{z_2}} = \hat{\mathbf{y}} - \mathbf{y}
   $$

2. **隐层到输入层**：
   - 计算损失对隐层激活值的梯度：
   $$
   \frac{\partial L}{\partial \mathbf{a_1}} = \mathbf{W_2}^T \frac{\partial L}{\partial \mathbf{z_2}}
   $$
   - 计算损失对隐层未激活值的梯度（ReLU激活函数）：
   $$
   \frac{\partial L}{\partial \mathbf{z_1}} = \frac{\partial L}{\partial \mathbf{a_1}} \od \text{ReLU}'(\mathbf{z_1})
   $$
   其中，$\text{ReLU}'(\mathbf{z_1})$ 是ReLU激活函数的导数，即：
   $$
   \text{ReLU}'(z) = \begin{cases} 
   1 & \text{if } z > 0 \\
   0 & \text{if } z \leq 0 
   \end{cases}
   $$
   - 计算损失对输入层权重和偏置的梯度：
   $$
   \frac{\partial L}{\partial \mathbf{W_1}} = \frac{\partial L}{\partial \mathbf{z_1}} \cdot \mathbf{x}^T
   $$
   $$
   \frac{\partial L}{\partial \mathbf{b_1}} = \frac{\partial L}{\partial \mathbf{z_1}}
   $$

#### 梯度更新

根据反向传播计算出的梯度，使用梯度下降法更新权重和偏置：
$$
\mathbf{W_1} \leftarrow \mathbf{W_1} - \eta \frac{\partial L}{\partial \mathbf{W_1}}
$$
$$
\mathbf{b_1} \leftarrow \mathbf{b_1} - \eta \frac{\partial L}{\partial \mathbf{b_1}}
$$
$$
\mathbf{W_2} \leftarrow \mathbf{W_2} - \eta \frac{\partial L}{\partial \mathbf{W_2}}
$$
$$
\mathbf{b_2} \leftarrow \mathbf{b_2} - \eta \frac{\partial L}{\partial \mathbf{b_2}}
$$

### 总结

1. **前向传播**：计算隐层未激活值 $\mathbf{z_1}$、隐层激活值 $\mathbf{a_1}$，以及输出层未激活值 $\mathbf{z_2}$。
2. **计算损失**：使用MSE损失函数计算输出 $\hat{\mathbf{y}}$ 与真实值 $\mathbf{y}$ 之间的损失。
3. **反向传播**：通过链式法则，从输出层开始，逐层计算梯度，并更新各层的权重和偏置。

通过上述步骤，可以推导出一层隐层的神经网络使用ReLU激活函数和MSE损失函数的反向传播过程。这一过程在深度学习中至关重要，因为它确保了模型能够通过训练数据逐步优化，从而提高预测性能。