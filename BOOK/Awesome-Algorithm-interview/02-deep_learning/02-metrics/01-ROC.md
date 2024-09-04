## ROC 曲线及其绘制方法

### 一、ROC 曲线的定义

ROC（Receiver Operating Characteristic）曲线是一种用来评估二分类模型性能的工具。ROC 曲线绘制的是在不同阈值下，模型的 **真正例率**（True Positive Rate, TPR） 和 **假正例率**（False Positive Rate, FPR）的关系。

### 二、ROC 曲线的坐标轴

1. **横轴（X-axis）**：假正例率（False Positive Rate, FPR）
   - 计算公式：$ \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} $
   - 其中，FP（False Positive）是假正例数量，TN（True Negative）是真负例数量。

2. **纵轴（Y-axis）**：真正例率（True Positive Rate, TPR）
   - 计算公式：$ \text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} $
   - 其中，TP（True Positive）是真正例数量，FN（False Negative）是假负例数量。

### 三、ROC 曲线的绘制步骤

1. **计算不同阈值下的 TPR 和 FPR**：
   - 将预测概率从高到低排序，设定多个阈值（如从0到1）。
   - 对每个阈值，计算对应的 TPR 和 FPR。
   - 记录所有阈值下的 TPR 和 FPR。

2. **绘制 ROC 曲线**：
   - 以 FPR 为横轴，TPR 为纵轴，在坐标系中绘制点。
   - 连接所有点，形成 ROC 曲线。

3. **计算 AUC（Area Under Curve）**：
   - AUC 值表示 ROC 曲线下的面积，数值范围在0到1之间。
   - AUC 越接近1，模型性能越好。AUC 为0.5 表示模型性能与随机猜测相同。

### 四、实例

假设我们有一个二分类模型的预测结果和实际标签：

```python
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 假设的预测概率和实际标签
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])

# 计算 FPR 和 TPR
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# 计算 AUC
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
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
```

在这个示例中，`roc_curve` 函数计算不同阈值下的 FPR 和 TPR，`auc` 函数计算 AUC 值，最后使用 `matplotlib` 绘制 ROC 曲线。

### 五、总结

ROC 曲线是评估二分类模型性能的重要工具，通过绘制不同阈值下的 TPR 和 FPR，可以全面了解模型的分类能力。AUC 值则提供了一个量化的指标，用于比较不同模型的优劣。

### 参考资料

- [Scikit-learn documentation on ROC curve](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
- [Understanding ROC curves and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)