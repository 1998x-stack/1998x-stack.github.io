> 旋转打印数组


旋转打印数组（螺旋矩阵）是一种经典的算法问题。给定一个二维数组（矩阵），按照顺时针螺旋顺序打印出矩阵的所有元素。

### 方法：模拟螺旋顺序

为了模拟螺旋顺序，我们可以依次按照上、右、下、左的顺序遍历矩阵的边界，并逐步缩小边界，直到所有元素都被打印。

### 代码实现

以下是一个 Python 实现螺旋矩阵打印的代码：

```python
def spiral_order(matrix):
    result = []
    if not matrix:
        return result
    
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # 从左到右遍历上边界
        for i in range(left, right + 1):
            result.append(matrix[top][i])
        top += 1
        
        # 从上到下遍历右边界
        for i in range(top, bottom + 1):
            result.append(matrix[right][i])
        right -= 1
        
        if top <= bottom:
            # 从右到左遍历下边界
            for i in range(right, left - 1, -1):
                result.append(matrix[bottom][i])
            bottom -= 1
        
        if left <= right:
            # 从下到上遍历左边界
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
            
    return result

# 示例使用
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
print(spiral_order(matrix))  # 输出: [1, 2, 3, 6, 9, 8, 7, 4, 5]
```

### 解释

1. **初始化边界**：
   - `top` 和 `bottom` 分别表示矩阵的上边界和下边界。
   - `left` 和 `right` 分别表示矩阵的左边界和右边界。

2. **遍历矩阵**：
   - 按照顺时针方向依次遍历上、右、下、左四个边界，并逐步缩小边界。
   - 从左到右遍历上边界，并将 `top` 增加 1。
   - 从上到下遍历右边界，并将 `right` 减少 1。
   - 如果 `top <= bottom`，从右到左遍历下边界，并将 `bottom` 减少 1。
   - 如果 `left <= right`，从下到上遍历左边界，并将 `left` 增加 1。

3. **终止条件**：
   - 当 `top > bottom` 或 `left > right` 时，所有元素都被遍历，循环结束。

### 时间复杂度和空间复杂度

- **时间复杂度**：O(m * n)，其中 m 和 n 分别为矩阵的行数和列数。每个元素被访问一次。
- **空间复杂度**：O(1)，不考虑结果列表的存储空间，只使用了常数额外空间。

### 参考文献

1. **LeetCode Problem - Spiral Matrix**：
   - 提供了螺旋矩阵问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/spiral-matrix/)

2. **GeeksforGeeks - Print a given matrix in spiral form**：
   - 提供了详细的螺旋矩阵打印方法和示例。
   - [链接](https://www.geeksforgeeks.org/print-a-given-matrix-in-spiral-form/)