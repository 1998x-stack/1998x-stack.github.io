> 详细手推SVM


### 手推支持向量机（SVM）

支持向量机（Support Vector Machine, SVM）是一种用于分类任务的监督学习算法，其核心思想是找到一个最优超平面，使得超平面两侧的样本间隔最大，从而实现分类。SVM可以用于线性可分和非线性可分的数据集。

#### 1. 线性可分SVM

**目标**：
找到一个超平面将数据集正确分类，并使得分类间隔最大化。超平面的方程为：
$$ \mathbf{w}^T \mathbf{x} + b = 0 $$

**约束条件**：
对于每个训练样本 $(\mathbf{x}_i, y_i)$，其中 $y_i \in \{-1, 1\}$，满足以下约束：
$$ y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 $$

**优化目标**：
最大化间隔等价于最小化 $ \|\mathbf{w}\|^2 $：
$$ \min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 $$

**拉格朗日函数**：
引入拉格朗日乘子 $\alpha_i $，构建拉格朗日函数：
$$ L(\mathbf{w}, b, \alpha) = \frac{1}{2} \|\mathbf{w}\|^2 - \sum_{i=1}^{m} \alpha_i [y_i (\mathbf{w}^T \mathbf{x}_i + b) - 1] $$

**KKT条件**：
根据KKT条件，求导并设置导数为零：
$$ \frac{\partial L}{\partial \mathbf{w}} = \mathbf{w} - \sum_{i=1}^{m} \alpha_i y_i \mathbf{x}_i = 0 $$
$$ \mathbf{w} = \sum_{i=1}^{m} \alpha_i y_i \mathbf{x}_i $$

$$ \frac{\partial L}{\partial b} = \sum_{i=1}^{m} \alpha_i y_i = 0 $$

**对偶问题**：
将以上结果代入拉格朗日函数，得到对偶问题：
$$ \max_{\alpha} \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j $$
$$ \text{subject to } \sum_{i=1}^{m} \alpha_i y_i = 0 \text{ and } \alpha_i \geq 0 $$

**决策函数**：
通过求解对偶问题得到拉格朗日乘子 $\alpha_i$，进而得到模型参数 $\mathbf{w}$ 和 $b$。最终的分类决策函数为：
$$ f(\mathbf{x}) = \text{sign}(\sum_{i=1}^{m} \alpha_i y_i \mathbf{x}_i^T \mathbf{x} + b) $$

#### 2. 非线性可分SVM

对于非线性可分数据，SVM通过核技巧（Kernel Trick）将输入数据映射到高维空间，使得在高维空间中线性可分。

**核函数**：
常用的核函数包括：
- 多项式核（Polynomial Kernel）：$$ K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + c)^d $$
- 高斯核（RBF Kernel）：$$ K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2) $$

通过使用核函数，优化问题变为：
$$ \max_{\alpha} \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) $$

**决策函数**：
非线性SVM的决策函数为：
$$ f(\mathbf{x}) = \text{sign}(\sum_{i=1}^{m} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b) $$

### 参考文献

1. **Pattern Recognition and Machine Learning by Christopher M. Bishop**：
   - 提供了支持向量机的详细理论和推导。
   - [书籍链接](https://www.springer.com/gp/book/9780387310732)

2. **The Elements of Statistical Learning by Trevor Hastie, Robert Tibshirani, and Jerome Friedman**：
   - 介绍了SVM及其他机器学习算法。
   - [书籍链接](https://web.stanford.edu/~hastie/ElemStatLearn/)

3. **An Introduction to Support Vector Machines and Other Kernel-based Learning Methods by Nello Cristianini and John Shawe-Taylor**：
   - 详细讨论了支持向量机和核方法。
   - [书籍链接](https://www.cambridge.org/core/books/an-introduction-to-support-vector-machines-and-other-kernel-based-learning-methods/B5CE6D8A8D9B9474AB899CF0AE8987BC)