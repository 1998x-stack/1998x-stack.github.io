## 常见的损失函数和激活函数

### 一、损失函数（Loss Function）

损失函数是用于评估模型预测结果与真实结果之间差异的函数。以下是几种常见的损失函数：

1. **均方误差（Mean Squared Error, MSE）**
$ L(\mathbf{y}, \mathbf{t}) = \frac{1}{n} \sum_{i=1}^n (y_i - t_i)^2 $
   - 适用于回归问题
   - 优点：易于计算和求导
   - 缺点：对异常值敏感

2. **交叉熵损失（Cross-Entropy Loss）**
$ L(\mathbf{y}, \mathbf{t}) = -\sum_{i=1}^n t_i \log y_i $
   - 适用于分类问题，尤其是多分类问题
   - 优点：能有效衡量概率分布之间的差异
   - 缺点：对标签进行独热编码处理

3. **二元交叉熵损失（Binary Cross-Entropy Loss）**
$ L(\mathbf{y}, \mathbf{t}) = -\frac{1}{n} \sum_{i=1}^n [t_i \log y_i + (1 - t_i) \log (1 - y_i)] $
   - 适用于二分类问题
   - 优点：适合输出为概率的模型

### 二、激活函数（Activation Function）

激活函数决定了神经元的输出。常见的激活函数有：

1. **Sigmoid**
$ \sigma(z) = \frac{1}{1 + e^{-z}} $
   - 优点：输出范围在 (0, 1) 之间，适用于概率预测
   - 缺点：梯度消失问题，收敛慢

2. **Softmax**
$ \sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} $
   - 优点：输出为概率分布，适用于多分类问题
   - 缺点：计算复杂度高

3. **ReLU（Rectified Linear Unit）**
$ \sigma(z) = \max(0, z) $
   - 优点：计算简单，不会出现梯度消失问题
   - 缺点：在负半轴上输出恒为0，可能导致神经元死亡

4. **tanh**
$ \sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} $
   - 优点：输出范围在 (-1, 1) 之间，收敛速度快于Sigmoid
   - 缺点：仍存在梯度消失问题

### 三、常见激活函数之间的关系及优势

1. **Sigmoid 和 Softmax 的关系**
   - 都可以用于分类问题
   - Sigmoid 常用于二分类问题，输出一个概率
   - Softmax 常用于多分类问题，输出一个概率分布

2. **ReLU 对比 tanh 的优势**
   - ReLU 在正轴上保持线性，不会出现梯度消失问题
   - ReLU 计算速度快，简单有效
   - tanh 虽然在范围上更合理（-1 到 1），但仍有梯度消失问题，尤其是在深层网络中

### 四、举例

假设我们有一个两分类问题，输入为 $ \mathbf{x} $，输出为 $ \mathbf{y} $，标签为 $ \mathbf{t} $。我们选择使用 Sigmoid 激活函数和二元交叉熵损失函数。

1. **前向传播：**
$ z = \mathbf{W} \mathbf{x} + \mathbf{b} $
$ y = \sigma(z) = \frac{1}{1 + e^{-z}} $

2. **损失函数：**
$ L(y, t) = -[t \log y + (1 - t) \log (1 - y)] $

3. **反向传播：**
$ \delta = y - t $
$ \frac{\partial L}{\partial \mathbf{W}} = \delta \mathbf{x}^T $
$ \frac{\partial L}{\partial \mathbf{b}} = \delta $

通过上述步骤，可以清晰地理解常见损失函数和激活函数的使用及其优势。