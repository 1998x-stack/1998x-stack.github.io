> L1和L2正则分别有什么特点？为何L1稀疏？

### L1 和 L2 正则化的特点

#### L1 正则化（Lasso）

**定义**：
L1 正则化在损失函数中加入权重绝对值的和，其公式如下：
$$ L_{\text{L1}} = \sum_{i=1}^n |w_i| $$

**特点**：
1. **稀疏性**：L1 正则化会产生稀疏解，即一些权重会被缩减为零。这样可以进行特征选择，使模型更加简洁。
2. **鲁棒性**：对异常值有较好的鲁棒性，因为它对大误差进行较少的惩罚。
3. **适用场景**：在需要特征选择的高维数据集中，L1 正则化非常有效，如文本分类、基因数据分析等。

**为何 L1 稀疏**：
- L1 正则化的目标是最小化权重的绝对值和，这导致了一些权重被压缩到零。
- 当梯度下降时，L1 正则化对权重的更新方向是固定的，这意味着它更倾向于使较小的权重迅速衰减到零，产生稀疏的权重向量。

#### L2 正则化（Ridge）

**定义**：
L2 正则化在损失函数中加入权重平方和，其公式如下：
$$ L_{\text{L2}} = \sum_{i=1}^n w_i^2 $$

**特点**：
1. **平滑性**：L2 正则化会使权重趋向于更小且均匀的值，而不会使它们变为零。
2. **防止过拟合**：通过惩罚大权重，L2 正则化有效地防止模型过拟合。
3. **适用场景**：在大部分机器学习任务中，L2 正则化都非常有效，尤其是当所有特征都可能有贡献时，如回归分析、图像分类等。

### L1 和 L2 正则化的比较

1. **稀疏性**：
   - **L1 正则化**：能够产生稀疏解，适合特征选择。
   - **L2 正则化**：不会产生稀疏解，权重趋向于更小且均匀的值。

2. **解决问题的类型**：
   - **L1 正则化**：适用于高维数据和需要特征选择的情况。
   - **L2 正则化**：适用于防止过拟合，并保持所有特征的贡献。

3. **优化方式**：
   - **L1 正则化**：可能导致优化问题变得非平滑，梯度下降法可能不如 L2 高效。
   - **L2 正则化**：优化问题是平滑的，梯度下降法收敛速度较快。

### 公式总结

- **L1 正则化**：
  $$ J(\theta) = J_{\text{original}}(\theta) + \lambda \sum_{i=1}^n |\theta_i| $$

- **L2 正则化**：
  $$ J(\theta) = J_{\text{original}}(\theta) + \lambda \sum_{i=1}^n \theta_i^2 $$

### 参考资料

1. **"Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville**: This book provides a comprehensive explanation of regularization techniques including L1 and L2 regularization.
   - [Book link](http://www.deeplearningbook.org/)

2. **"Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman**: This book offers an in-depth discussion on L1 (Lasso) and L2 (Ridge) regularization.
   - [Book link](https://web.stanford.edu/~hastie/ElemStatLearn/)

3. **Understanding L1 and L2 Regularization**:
   - [Towards Data Science article](https://towardsdatascience.com/understanding-l1-and-l2-regularization-7f1b4c21fb7e)
   - [Analytics Vidhya article](https://www.analyticsvidhya.com/blog/2021/01/understanding-the-difference-between-l1-and-l2-regularization/)