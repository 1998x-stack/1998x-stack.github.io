> 行列递增矩阵查找数字


在一个行列递增矩阵中查找目标值是一个经典的问题，通常可以通过有效的算法快速找到目标值。行列递增矩阵的特点是每一行和每一列都是递增的。

### 方法：从右上角开始搜索

从右上角开始搜索是一种高效的方法。因为在右上角的位置：
- 如果当前值比目标值大，可以排除当前列。
- 如果当前值比目标值小，可以排除当前行。

### 算法步骤

1. 初始化行索引 `row` 为 0，列索引 `col` 为矩阵的最后一列索引。
2. 循环直到 `row` 超过矩阵的行数或 `col` 小于 0：
   - 如果矩阵元素 `matrix[row][col]` 等于目标值，则返回 `[row, col]`。
   - 如果 `matrix[row][col]` 大于目标值，向左移动一列（`col -= 1`）。
   - 如果 `matrix[row][col]` 小于目标值，向下移动一行（`row += 1`）。
3. 如果循环结束还没有找到目标值，返回 `[-1, -1]` 表示目标值不在矩阵中。

### 代码实现

```python
def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return [-1, -1]

    rows, cols = len(matrix), len(matrix[0])
    row, col = 0, cols - 1

    while row < rows and col >= 0:
        if matrix[row][col] == target:
            return [row, col]
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1

    return [-1, -1]

# 示例使用
matrix = [
    [1, 4, 7, 11, 15],
    [2, 5, 8, 12, 19],
    [3, 6, 9, 16, 22],
    [10, 13, 14, 17, 24],
    [18, 21, 23, 26, 30]
]
target = 5
print(search_matrix(matrix, target))  # 输出: [1, 1]

target = 20
print(search_matrix(matrix, target))  # 输出: [-1, -1]
```

### 解释

1. **初始化**：
   - `row` 设置为 0，从第一行开始。
   - `col` 设置为 `cols - 1`，从最后一列开始。

2. **循环搜索**：
   - 比较当前元素与目标值。
   - 如果相等，返回当前索引 `[row, col]`。
   - 如果当前值大于目标值，向左移动一列。
   - 如果当前值小于目标值，向下移动一行。

3. **结束条件**：
   - 当 `row` 超过矩阵的行数或 `col` 小于 0 时，搜索结束，返回 `[-1, -1]` 表示未找到目标值。

### 时间复杂度和空间复杂度

- **时间复杂度**：O(m + n)，其中 m 为矩阵的行数，n 为矩阵的列数。每次移动都会排除一行或一列，因此最多需要 m + n 步。
- **空间复杂度**：O(1)，只使用了常数额外空间。

### 参考文献

1. **GeeksforGeeks - Search in a row wise and column wise sorted matrix**：
   - 提供了详细的行列递增矩阵查找方法和实现。
   - [链接](https://www.geeksforgeeks.org/search-in-row-wise-and-column-wise-sorted-matrix/)

2. **LeetCode Problem - Search a 2D Matrix II**：
   - 提供了行列递增矩阵查找问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/search-a-2d-matrix-ii/)