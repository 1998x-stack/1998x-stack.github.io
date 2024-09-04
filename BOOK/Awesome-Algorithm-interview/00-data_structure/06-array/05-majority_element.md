> 查找出现超过一半的数字


查找数组中出现次数超过一半的数字（多数元素）是一类经典问题，可以使用多种算法来解决，其中最常见和高效的是摩尔投票算法（Boyer-Moore Voting Algorithm）。

### 方法：摩尔投票算法

摩尔投票算法是一种线性时间复杂度 $O(n)$ 和常数空间复杂度 $O(1)$ 的算法，适用于寻找多数元素。该算法基于以下原理：如果一个元素出现次数超过数组长度的一半，那么它一定会在数组的中位数位置出现。

#### 算法步骤

1. **初始化候选人和计数器**：
   - 设置一个候选人变量 `candidate`，初始化为空。
   - 设置一个计数器 `count`，初始化为0。

2. **第一次遍历数组**：
   - 遍历数组中的每个元素 `num`：
     - 如果 `count` 为0，则将 `candidate` 设置为当前元素 `num`。
     - 如果 `num` 与 `candidate` 相同，则将 `count` 增加1。
     - 如果 `num` 与 `candidate` 不同，则将 `count` 减少1。

3. **第二次遍历数组（可选）**：
   - 确认 `candidate` 是多数元素：
     - 遍历数组，计算 `candidate` 的实际出现次数，确保其确实超过数组长度的一半。

### 代码实现

```python
def majority_element(nums):
    # 第一遍遍历寻找候选人
    candidate = None
    count = 0
    for num in nums:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)

    # 第二遍遍历确认候选人
    count = 0
    for num in nums:
        if num == candidate:
            count += 1

    if count > len(nums) // 2:
        return candidate
    else:
        return None

# 示例使用
nums = [2, 2, 1, 1, 1, 2, 2]
print(majority_element(nums))  # 输出: 2
```

### 解释

1. **第一次遍历**：
   - 初始化 `candidate` 和 `count`。
   - 遍历数组，当 `count` 为0时更新 `candidate`。
   - 如果当前元素与 `candidate` 相同，增加 `count`；否则，减少 `count`。

2. **第二次遍历（可选）**：
   - 确认 `candidate` 是否确实为多数元素，遍历数组计算 `candidate` 的实际出现次数。
   - 如果 `candidate` 的出现次数超过数组长度的一半，返回 `candidate`；否则，返回 `None`。

### 优点和缺点

**优点**：
- **时间复杂度**：$O(n)$，只需两次遍历数组。
- **空间复杂度**：$O(1)$，只使用常数额外空间。

**缺点**：
- 需要保证输入数组中确实存在多数元素，否则需要进行额外的确认步骤。

### 参考文献

1. **GeeksforGeeks - Majority Element**：
   - 提供了摩尔投票算法的详细解释和代码实现。
   - [链接](https://www.geeksforgeeks.org/majority-element/)
   
2. **LeetCode Problem - Majority Element**：
   - 提供了多数元素问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/majority-element/)