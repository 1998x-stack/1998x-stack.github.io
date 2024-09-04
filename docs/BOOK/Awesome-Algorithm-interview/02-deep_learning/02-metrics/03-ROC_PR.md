### ROC曲线（Receiver Operating Characteristic Curve）

#### 定义
ROC曲线是评估二分类模型性能的一种工具，通过在不同阈值下计算模型的**真正例率（True Positive Rate, TPR）**和**假正例率（False Positive Rate, FPR）**，来绘制出TPR对FPR的曲线。

#### 关键点
1. **横轴（X-axis）**：假正例率（FPR）
   - 计算公式：$$ \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} $$
   - FP是假正例数量，TN是真负例数量。

2. **纵轴（Y-axis）**：真正例率（TPR）
   - 计算公式：$$ \text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} $$
   - TP是真正例数量，FN是假负例数量。

3. **AUC（Area Under Curve）**：曲线下面积，AUC值越大，模型性能越好。AUC值为0.5表示模型性能与随机猜测相同。

#### 适用场景
ROC曲线适用于处理类别分布相对均衡的数据集，因为它考虑了所有类别的情况，对类别不平衡问题不敏感。

### PR曲线（Precision-Recall Curve）

#### 定义
PR曲线通过在不同阈值下计算模型的**精确率（Precision）**和**召回率（Recall）**，来绘制出Precision对Recall的曲线。

#### 关键点
1. **横轴（X-axis）**：召回率（Recall）
   - 计算公式：$$ \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} $$

2. **纵轴（Y-axis）**：精确率（Precision）
   - 计算公式：$$ \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} $$

3. **AP（Average Precision）**：PR曲线下面积，表示模型在不同阈值下的平均性能。

#### 适用场景
PR曲线在处理类别不平衡的数据集时更加有效，因为它关注正类（少数类）的表现。

### 比较与总结

1. **ROC曲线**：
   - 优点：对阈值选择不敏感，适用于类别分布相对均衡的数据集。
   - 缺点：在类别严重不平衡时，ROC曲线可能会给出过于乐观的评估。

2. **PR曲线**：
   - 优点：在类别不平衡的数据集中表现更好，因为它关注正类的表现。
   - 缺点：在类别分布均衡时可能没有ROC曲线直观。

### 示例代码

下面是使用Python绘制ROC曲线和PR曲线的示例代码：

```python
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# 假设的预测概率和实际标签
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])

# ROC曲线和AUC
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# PR曲线和平均精度
precision, recall, _ = precision_recall_curve(y_true, y_scores)
average_precision = average_precision_score(y_true, y_scores)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 绘制PR曲线
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.show()
```

### 参考资料
- [Scikit-learn Documentation on ROC and PR curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
- [Understanding ROC and PR curves](https://towardsdatascience.com/understanding-roc-curves-and-pr-curves-a8b561e515af)