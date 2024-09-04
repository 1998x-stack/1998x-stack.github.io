> 数组、二分、快排系列：旋转数组查找



### 在旋转数组中查找目标值

旋转数组是通过将有序数组的某个前缀移到数组的末尾形成的。例如，有序数组 `[0, 1, 2, 4, 5, 6, 7]` 可以被旋转形成 `[4, 5, 6, 7, 0, 1, 2]`。查找目标值可以通过修改后的二分查找算法实现。

### 二分查找算法的基本思想

二分查找算法通过反复将查找范围减半，从而实现高效查找。在旋转数组中，由于数组的局部有序性，二分查找算法需要进行一些调整。

### 算法步骤

1. **初始化**：
   - 设置两个指针 `left` 和 `right`，分别指向数组的起始和结束位置。

2. **循环查找**：
   - 计算中间位置 `mid`。
   - 判断目标值是否等于中间值 `nums[mid]`，如果相等，则返回 `mid`。
   - 判断左半部分是否有序，即 `nums[left] <= nums[mid]`：
     - 如果左半部分有序，且目标值在左半部分范围内，即 `nums[left] <= target < nums[mid]`，则在左半部分查找，将 `right` 更新为 `mid - 1`。
     - 否则在右半部分查找，将 `left` 更新为 `mid + 1`。
   - 如果右半部分有序，即 `nums[mid] < nums[right]`：
     - 如果目标值在右半部分范围内，即 `nums[mid] < target <= nums[right]`，则在右半部分查找，将 `left` 更新为 `mid + 1`。
     - 否则在左半部分查找，将 `right` 更新为 `mid - 1`。

3. **返回结果**：
   - 如果循环结束且未找到目标值，则返回 -1。

### 代码实现

```python
def search_rotated_array(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid

        # 左半部分有序
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # 右半部分有序
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1

# 示例使用
nums = [4, 5, 6, 7, 0, 1, 2]
target = 0
print(search_rotated_array(nums, target))  # 输出: 4

target = 3
print(search_rotated_array(nums, target))  # 输出: -1
```

### 解释

1. **初始化**：
   - `left` 和 `right` 分别指向数组的起始和结束位置。

2. **循环查找**：
   - 通过计算中间位置 `mid`，并判断中间值是否为目标值。
   - 根据数组的局部有序性，确定查找范围在左半部分还是右半部分。

3. **返回结果**：
   - 如果找到目标值，返回其位置。
   - 如果未找到目标值，返回 -1。

### 优点和缺点

**优点**：
- 时间复杂度为 O(log n)，具有较高的查找效率。
- 适用于大型数组和需要频繁查找的场景。

**缺点**：
- 需要数组预先进行旋转，且要求数组元素无重复。

### 参考文献

1. **GeeksforGeeks - Search an element in a sorted and rotated array**：
   - 提供了详细的旋转数组查找算法及其实现。
   - [链接](https://www.geeksforgeeks.org/search-an-element-in-a-sorted-and-pivoted-array/)

2. **LeetCode - Search in Rotated Sorted Array**：
   - 提供了旋转数组查找问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/search-in-rotated-sorted-array/)