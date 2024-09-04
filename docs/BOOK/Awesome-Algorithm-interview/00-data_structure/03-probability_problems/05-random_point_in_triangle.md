> 三角形内部等概率随机选取一个点

在三角形内部等概率随机选取一个点是一项有趣的几何概率问题。为了确保点在三角形内部均匀分布，我们可以使用重心坐标（Barycentric coordinates）进行采样。这种方法简单且易于实现。

### 方法：重心坐标法

假设我们有一个三角形，其顶点分别为 $A (x_1, y_1)$、$B (x_2, y_2)$、$C (x_3, y_3)$。要在这个三角形内随机选取一个点，我们可以按以下步骤进行：

1. 生成两个在区间 $[0, 1]$ 内的随机数 $u$ 和 $v$。
2. 如果 $u + v > 1$，则重新生成 $u$ 和 $v$。
3. 用重心坐标公式计算随机点的坐标：
   $$
   x = (1 - \sqrt{u}) x_1 + \sqrt{u} (1 - v) x_2 + \sqrt{u} v x_3
   $$
   $$
   y = (1 - \sqrt{u}) y_1 + \sqrt{u} (1 - v) y_2 + \sqrt{u} v y_3
   $$

### 代码实现

下面是使用Python实现这一方法的代码：

```python
import random

def random_point_in_triangle(A, B, C):
    while True:
        u = random.random()
        v = random.random()
        if u + v <= 1:
            break
    
    sqrt_u = u ** 0.5
    one_minus_sqrt_u = 1 - sqrt_u
    one_minus_v = 1 - v
    
    x = one_minus_sqrt_u * A[0] + sqrt_u * one_minus_v * B[0] + sqrt_u * v * C[0]
    y = one_minus_sqrt_u * A[1] + sqrt_u * one_minus_v * B[1] + sqrt_u * v * C[1]
    
    return (x, y)

# 示例使用
A = (0, 0)
B = (1, 0)
C = (0, 1)

# 生成一个随机点
point = random_point_in_triangle(A, B, C)
print(point)
```

### 解释

1. **生成随机数**：
   - 我们使用 `random.random()` 生成两个 $[0, 1]$ 之间的随机数 $u$ 和 $v$。
   - 确保 $u + v \leq 1$ 以保证点在三角形内部。如果 $u + v > 1$，则重新生成。

2. **计算点的坐标**：
   - 使用重心坐标公式将随机数 $u$ 和 $v$ 转换为三角形内的随机点的坐标。

### 优点

- **均匀分布**：通过重心坐标法生成的点在三角形内是均匀分布的。
- **简单易实现**：代码实现简洁，计算量小。

### 参考文献

1. **GeeksforGeeks - Random point inside a Triangle**：
   - 详细介绍了如何在三角形内部随机选点，并提供了多种实现方法。
   - [链接](https://www.geeksforgeeks.org/random-point-inside-a-triangle/)
   
2. **Wikipedia - Barycentric coordinates (mathematics)**：
   - 提供了关于重心坐标的详细解释和应用。
   - [链接](https://en.wikipedia.org/wiki/Barycentric_coordinates_(mathematics))
