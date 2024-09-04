> 查找和为定值的两个数


查找和为定值的两个数是一个常见的编程问题，通常称为“两数之和”（Two Sum）问题。可以使用多种方法来解决这个问题，其中最常见和高效的方法是哈希表。

### 方法一：使用哈希表

哈希表（Hash Table）是一种高效的数据结构，可以在常数时间内完成插入和查找操作。通过使用哈希表，我们可以在一次遍历数组的过程中找到和为定值的两个数。

#### 算法步骤

1. 初始化一个空的哈希表。
2. 遍历数组，对于每个元素 `num`：
   - 计算目标值 `target` 和当前元素 `num` 的差值 `complement`。
   - 检查 `complement` 是否存在于哈希表中：
     - 如果存在，说明找到了这两个数，返回它们的索引。
     - 如果不存在，将当前元素 `num` 及其索引存入哈希表。

### 代码实现

```python
def two_sum(nums, target):
    num_to_index = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i
    return []

# 示例使用
nums = [2, 7, 11, 15]
target = 9
print(two_sum(nums, target))  # 输出: [0, 1]
```

### 解释

1. **初始化**：
   - 创建一个空的哈希表 `num_to_index`，用于存储每个元素及其索引。

2. **遍历数组**：
   - 对于每个元素 `num`，计算目标值 `target` 与当前元素 `num` 的差值 `complement`。
   - 检查 `complement` 是否存在于哈希表 `num_to_index` 中。
     - 如果存在，返回 `complement` 和当前元素 `num` 的索引。
     - 如果不存在，将当前元素 `num` 及其索引存入哈希表 `num_to_index`。

### 优点和缺点

**优点**：
- 时间复杂度为 $O(n)$，其中 $n$ 是数组的长度。每个元素最多只需查找和插入一次。
- 空间复杂度为 $O(n)$，用于存储每个元素及其索引的哈希表。

**缺点**：
- 需要额外的哈希表存储空间。

### 方法二：排序和双指针

如果数组可以被修改，可以先对数组进行排序，然后使用双指针技术找到和为定值的两个数。

#### 算法步骤

1. 对数组进行排序。
2. 初始化两个指针 `left` 和 `right`，分别指向数组的起始和末尾。
3. 循环直到 `left` 小于 `right`：
   - 计算两个指针指向的元素之和 `sum`。
   - 如果 `sum` 等于目标值 `target`，返回这两个数的索引。
   - 如果 `sum` 小于 `target`，将 `left` 指针右移。
   - 如果 `sum` 大于 `target`，将 `right` 指针左移。

### 代码实现

```python
def two_sum_sorted(nums, target):
    nums_sorted = sorted(enumerate(nums), key=lambda x: x[1])
    left, right = 0, len(nums) - 1

    while left < right:
        sum = nums_sorted[left][1] + nums_sorted[right][1]
        if sum == target:
            return [nums_sorted[left][0], nums_sorted[right][0]]
        elif sum < target:
            left += 1
        else:
            right -= 1
    return []

# 示例使用
nums = [2, 7, 11, 15]
target = 9
print(two_sum_sorted(nums, target))  # 输出: [0, 1]
```

### 解释

1. **排序**：
   - 对数组进行排序，并保留原始索引。
   
2. **双指针**：
   - 使用两个指针 `left` 和 `right`，分别指向数组的起始和末尾。
   - 根据两个指针指向的元素之和 `sum` 与目标值 `target` 的比较结果，移动 `left` 或 `right` 指针，直到找到和为定值的两个数。

### 优点和缺点

**优点**：
- 时间复杂度为 $O(n \log n)$，其中 $n$ 是数组的长度。排序需要 $O(n \log n)$，双指针查找需要 $O(n)$。
- 空间复杂度为 $O(1)$（不考虑排序后的临时数组）。

**缺点**：
- 数组需要进行排序，如果不能修改原数组，则需要额外的空间存储排序后的数组。

### 参考文献

1. **LeetCode Problem - Two Sum**：
   - 提供了两数之和问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/two-sum/)

2. **GeeksforGeeks - Find a pair with given sum in an array**：
   - 提供了详细的两数之和问题的解释和多种解决方法。
   - [链接](https://www.geeksforgeeks.org/find-a-pair-with-given-sum-in-an-array/)