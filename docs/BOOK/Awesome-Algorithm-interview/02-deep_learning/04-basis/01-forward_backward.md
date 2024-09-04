## 前向传播和反向传播

### 一、前向传播 (Forward Propagation)

给定一个简单的两层神经网络，输入为一个一维向量 $ \mathbf{x} $，网络的输出为 $ \mathbf{y} $。网络的结构如下：

- 输入层：$ \mathbf{x} \in \mathbb{R}^n $
- 隐藏层：$ \mathbf{h} \in \mathbb{R}^m $，激活函数为 $ \sigma $
- 输出层：$ \mathbf{y} \in \mathbb{R}^k $，激活函数为 $ \phi $

假设参数如下：
- 输入层到隐藏层的权重矩阵：$ \mathbf{W}_1 \in \mathbb{R}^{m \times n} $
- 输入层到隐藏层的偏置向量：$ \mathbf{b}_1 \in \mathbb{R}^m $
- 隐藏层到输出层的权重矩阵：$ \mathbf{W}_2 \in \mathbb{R}^{k \times m} $
- 隐藏层到输出层的偏置向量：$ \mathbf{b}_2 \in \mathbb{R}^k $

前向传播的步骤如下：

1. 计算隐藏层的输入：
$ \mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1 $

2. 计算隐藏层的输出：
$ \mathbf{h} = \sigma(\mathbf{z}_1) $

3. 计算输出层的输入：
$ \mathbf{z}_2 = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2 $

4. 计算输出层的输出：
$ \mathbf{y} = \phi(\mathbf{z}_2) $

### 二、反向传播 (Back Propagation)

为了最小化损失函数 $ L(\mathbf{y}, \mathbf{t}) $，需要通过反向传播计算梯度，并使用梯度下降更新权重。假设损失函数为均方误差 (MSE)，即：
$ L(\mathbf{y}, \mathbf{t}) = \frac{1}{2} \|\mathbf{y} - \mathbf{t}\|^2 $

反向传播的步骤如下：

1. 计算输出层的梯度：
$ \delta_2 = \frac{\partial L}{\partial \mathbf{z}_2} = (\mathbf{y} - \mathbf{t}) \odot \phi'(\mathbf{z}_2) $

2. 计算隐藏层的梯度：
$ \delta_1 = \frac{\partial L}{\partial \mathbf{z}_1} = (\mathbf{W}_2^T \delta_2) \odot \sigma'(\mathbf{z}_1) $

3. 计算权重和偏置的梯度：
$ \frac{\partial L}{\partial \mathbf{W}_2} = \delta_2 \mathbf{h}^T $
$ \frac{\partial L}{\partial \mathbf{b}_2} = \delta_2 $
$ \frac{\partial L}{\partial \mathbf{W}_1} = \delta_1 \mathbf{x}^T $
$ \frac{\partial L}{\partial \mathbf{b}_1} = \delta_1 $

### 三、权重更新

使用梯度下降更新权重和偏置：

$ \mathbf{W}_2 \leftarrow \mathbf{W}_2 - \eta \frac{\partial L}{\partial \mathbf{W}_2} $
$ \mathbf{b}_2 \leftarrow \mathbf{b}_2 - \eta \frac{\partial L}{\partial \mathbf{b}_2} $
$ \mathbf{W}_1 \leftarrow \mathbf{W}_1 - \eta \frac{\partial L}{\partial \mathbf{W}_1} $
$ \mathbf{b}_1 \leftarrow \mathbf{b}_1 - \eta \frac{\partial L}{\partial \mathbf{b}_1} $

其中，$ \eta $ 是学习率。

### 四、实例计算

假设输入向量 $ \mathbf{x} = [x_1, x_2]^T $，网络参数为：

$ \mathbf{W}_1 = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix}, \quad \mathbf{b}_1 = \begin{bmatrix} b_{11} \\ b_{12} \end{bmatrix} $
$ \mathbf{W}_2 = \begin{bmatrix} w_{31} & w_{32} \end{bmatrix}, \quad \mathbf{b}_2 = b_{21} $

假设激活函数 $ \sigma $ 和 $ \phi $ 为ReLU，即 $ \sigma(z) = \phi(z) = \max(0, z) $。

1. 计算隐藏层的输入：
$ \mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1 = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} b_{11} \\ b_{12} \end{bmatrix} $

2. 计算隐藏层的输出：
$ \mathbf{h} = \sigma(\mathbf{z}_1) = \begin{bmatrix} \max(0, w_{11}x_1 + w_{12}x_2 + b_{11}) \\ \max(0, w_{21}x_1 + w_{22}x_2 + b_{12}) \end{bmatrix} $

3. 计算输出层的输入：
$ \mathbf{z}_2 = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2 = \begin{bmatrix} w_{31} & w_{32} \end{bmatrix} \begin{bmatrix} h_1 \\ h_2 \end{bmatrix} + b_{21} $

4. 计算输出层的输出：
$ \mathbf{y} = \phi(\mathbf{z}_2) = \max(0, w_{31}h_1 + w_{32}h_2 + b_{21}) $

通过上述步骤，可以详细地计算出前向传播和反向传播的结果。