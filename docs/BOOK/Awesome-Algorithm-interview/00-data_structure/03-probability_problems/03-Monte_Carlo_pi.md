> 采样 计算 pi 的大小


计算圆周率 $\pi$ 的一个常用方法是蒙特卡罗方法。这种方法利用随机采样来估计 $\pi$ 的值。下面是详细的步骤和Python代码实现。

### 蒙特卡罗方法计算 $\pi$ 的步骤

1. **随机采样点**：在单位正方形 $[0,1] \times [0,1]$ 内生成大量随机点。
2. **计算点落在圆内的比例**：计算这些点中有多少落在以原点为圆心、半径为1的四分之一圆内。
3. **估算 $\pi$ 的值**：利用圆的面积和正方形面积的比值来估计 $\pi$。

由于四分之一圆的面积为 $\pi/4$，而单位正方形的面积为1，因此，四分之一圆内点的比例应接近 $\pi/4$。具体来说，如果在单位正方形内生成 $N$ 个点，其中有 $M$ 个点落在四分之一圆内，那么可以用下面的公式估算 $\pi$：

$$ \pi \approx 4 \times \frac{M}{N} $$

### Python代码实现

```python
import random

def estimate_pi(num_samples):
    inside_circle = 0

    for _ in range(num_samples):
        x = random.random()  # 生成 [0,1] 之间的随机数
        y = random.random()  # 生成 [0,1] 之间的随机数
        if x*x + y*y <= 1:
            inside_circle += 1
    
    return (inside_circle / num_samples) * 4

# 示例使用
num_samples = 1000000  # 采样点数量
estimated_pi = estimate_pi(num_samples)
print(f"估计的 π 值: {estimated_pi}")
```

### 解释

1. **生成随机点**：使用 `random.random()` 生成 $[0,1]$ 区间的随机浮点数作为点的坐标。
2. **判断点是否在圆内**：检查点 $(x, y)$ 是否满足 $x^2 + y^2 \leq 1$。
3. **计算 $\pi$ 的估计值**：统计在圆内的点数，然后用公式 $\pi \approx 4 \times \frac{M}{N}$ 计算 $\pi$。

### 结果与精度

由于蒙特卡罗方法的随机性，采样点数 $N$ 越多，估计值的精度越高。可以通过增加 `num_samples` 的值来提高估计 $\pi$ 的精度。例如，使用 1,000,000 个采样点通常能得到小数点后4位左右精度的 $\pi$ 值。

### 总结

蒙特卡罗方法是一种简单而有效的估算 $\pi$ 的方法，适用于计算机仿真和随机数生成。通过不断增加采样点的数量，可以提高估算的精度。