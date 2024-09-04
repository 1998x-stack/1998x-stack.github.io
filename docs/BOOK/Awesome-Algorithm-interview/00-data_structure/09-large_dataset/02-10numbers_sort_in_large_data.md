> 10亿个1-10的数字排序


对于排序10亿个范围在1到10之间的数字，以下是具体的步骤和详细的Python代码实现：

1. **理解问题**：
   - 由于数字范围很小（1到10），但数量巨大（10亿），可以使用计数排序（Counting Sort）这种时间复杂度为O(n)的算法。

2. **步骤展开**：
   - **计数**：首先统计每个数字出现的次数。
   - **重建数组**：根据计数结果重建排序后的数组。

### 步骤一：计数
```python
import numpy as np

def count_numbers(data: np.ndarray) -> dict:
    """统计每个数字出现的次数"""
    counts = np.zeros(10, dtype=int)  # 数组索引0-9分别对应数字1-10
    for number in data:
        counts[number - 1] += 1
    return counts
```

### 步骤二：重建数组
```python
def rebuild_sorted_array(counts: np.ndarray) -> np.ndarray:
    """根据计数结果重建排序后的数组"""
    sorted_array = np.zeros(np.sum(counts), dtype=int)
    index = 0
    for number, count in enumerate(counts, start=1):
        sorted_array[index:index+count] = number
        index += count
    return sorted_array
```

### 主函数
```python
def main(data: np.ndarray) -> np.ndarray:
    # 计数阶段
    counts = count_numbers(data)
    
    # 重建排序数组阶段
    sorted_array = rebuild_sorted_array(counts)
    
    return sorted_array

# 示例使用
data = np.random.randint(1, 11, size=10**9)  # 生成10亿个1到10的随机数字
sorted_data = main(data)
```

### 代码注释：
1. **count_numbers**：使用一个长度为10的数组`counts`来统计每个数字出现的次数，数组索引0-9分别对应数字1-10。
2. **rebuild_sorted_array**：根据`counts`数组重建排序后的数组。`sorted_array`的长度为10亿，使用计数结果逐步填充数组。
3. **main**：主函数，调用上述两个函数，完成排序任务。

### 代码详细解释：
1. **计数**：
   - 创建一个长度为10的数组`counts`，初始化为0。
   - 遍历数据，每遇到一个数字，就将相应索引位置的计数加1。例如，遇到数字1，`counts[0]`加1；遇到数字2，`counts[1]`加1，以此类推。

2. **重建排序数组**：
   - 创建一个长度为10亿的数组`sorted_array`，用于存储排序后的结果。
   - 遍历`counts`数组，根据每个计数的值，在`sorted_array`中填充对应的数字。例如，`counts[0]`为10万，表示数字1出现了10万次，则在`sorted_array`中填充10万个1。

### 结果输出：
由于数据量极大，建议将结果保存到文件或以其他形式存储，而非直接输出到控制台。

```python
np.savetxt('sorted_data.txt', sorted_data, fmt='%d')
```