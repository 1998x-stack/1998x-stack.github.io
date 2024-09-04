> 快排的变化


### 快速排序的变化和改进

快速排序（Quick Sort）是一种高效的比较排序算法，平均时间复杂度为 $O(n \log n)$。虽然它在大多数情况下表现优越，但最差情况下时间复杂度为 $O(n^2)$。为了提高性能并避免最差情况，快速排序有多种变化和改进版本。

#### 1. 三向切分快速排序（3-Way Quick Sort）

三向切分快速排序通过将数组分成三部分：小于基准值、等于基准值和大于基准值，从而有效处理包含大量重复元素的数组。

**算法步骤**：
1. 选择一个基准值 `pivot`。
2. 维护三个指针：`lt`（小于基准值部分的末尾）、`gt`（大于基准值部分的开始）和 `i`（当前元素）。
3. 遍历数组，将元素分成三部分：小于 `pivot`、等于 `pivot` 和大于 `pivot`。
4. 对小于 `pivot` 和大于 `pivot` 的部分递归进行三向切分快速排序。

**代码实现**：
```python
def three_way_quick_sort(arr, lo, hi):
    if lo >= hi:
        return
    lt, i, gt = lo, lo + 1, hi
    pivot = arr[lo]
    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[gt], arr[i] = arr[i], arr[gt]
            gt -= 1
        else:
            i += 1
    three_way_quick_sort(arr, lo, lt - 1)
    three_way_quick_sort(arr, gt + 1, hi)

# 示例使用
arr = [3, 5, 2, 1, 9, 7, 8, 4, 6, 0, 3, 7, 4]
three_way_quick_sort(arr, 0, len(arr) - 1)
print(arr)  # 输出: 排序后的数组
```

### 2. 随机快速排序（Randomized Quick Sort）

通过随机选择基准值来避免最差情况。这样可以减少基准值选择不当导致的最差时间复杂度。

**算法步骤**：
1. 在每次递归调用前随机选择一个基准值。
2. 将随机选择的基准值与当前区间的第一个元素交换。
3. 按照经典的快速排序算法进行排序。

**代码实现**：
```python
import random

def randomized_quick_sort(arr, lo, hi):
    if lo < hi:
        pivot_index = random_partition(arr, lo, hi)
        randomized_quick_sort(arr, lo, pivot_index - 1)
        randomized_quick_sort(arr, pivot_index + 1, hi)

def random_partition(arr, lo, hi):
    pivot_index = random.randint(lo, hi)
    arr[lo], arr[pivot_index] = arr[pivot_index], arr[lo]
    return partition(arr, lo, hi)

def partition(arr, lo, hi):
    pivot = arr[lo]
    i = lo
    for j in range(lo + 1, hi + 1):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i], arr[lo] = arr[lo], arr[i]
    return i

# 示例使用
arr = [3, 5, 2, 1, 9, 7, 8, 4, 6, 0]
randomized_quick_sort(arr, 0, len(arr) - 1)
print(arr)  # 输出: 排序后的数组
```

### 3. 双轴快速排序（Dual-Pivot Quick Sort）

双轴快速排序使用两个基准值将数组分成三个部分，从而优化性能。它在某些情况下比单轴快速排序更快。

**算法步骤**：
1. 选择两个基准值 `pivot1` 和 `pivot2`，并确保 `pivot1` 小于等于 `pivot2`。
2. 将数组分成三个部分：小于 `pivot1`、介于 `pivot1` 和 `pivot2` 之间、大于 `pivot2`。
3. 递归地对这三个部分进行排序。

**代码实现**：
```python
def dual_pivot_quick_sort(arr, low, high):
    if low < high:
        pivot1, pivot2 = partition(arr, low, high)
        dual_pivot_quick_sort(arr, low, pivot1 - 1)
        dual_pivot_quick_sort(arr, pivot1 + 1, pivot2 - 1)
        dual_pivot_quick_sort(arr, pivot2 + 1, high)

def partition(arr, low, high):
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
    pivot1 = arr[low]
    pivot2 = arr[high]
    i = low + 1
    lt = low + 1
    gt = high - 1

    while i <= gt:
        if arr[i] < pivot1:
            arr[i], arr[lt] = arr[lt], arr[i]
            lt += 1
        elif arr[i] > pivot2:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1
    lt -= 1
    gt += 1

    arr[low], arr[lt] = arr[lt], arr[low]
    arr[high], arr[gt] = arr[gt], arr[high]

    return lt, gt

# 示例使用
arr = [3, 5, 2, 1, 9, 7, 8, 4, 6, 0]
dual_pivot_quick_sort(arr, 0, len(arr) - 1)
print(arr)  # 输出: 排序后的数组
```

### 4. 三路分区快速排序（Three-Way Partitioning Quick Sort）

三路分区快速排序通过处理等于基准值的元素来优化快速排序，特别适用于包含大量重复元素的数组。

**算法步骤**：
1. 选择一个基准值 `pivot`。
2. 维护三个区间：小于 `pivot`、等于 `pivot` 和大于 `pivot`。
3. 将等于 `pivot` 的元素跳过，仅递归处理小于 `pivot` 和大于 `pivot` 的区间。

**代码实现**：
```python
def three_way_partition(arr, low, high):
    if low < high:
        lt, gt = low, high
        pivot = arr[low]
        i = low
        while i <= gt:
            if arr[i] < pivot:
                arr[lt], arr[i] = arr[i], arr[lt]
                lt += 1
                i += 1
            elif arr[i] > pivot:
                arr[gt], arr[i] = arr[i], arr[gt]
                gt -= 1
            else:
                i += 1
        three_way_partition(arr, low, lt - 1)
        three_way_partition(arr, gt + 1, high)

# 示例使用
arr = [3, 5, 2, 1, 9, 7, 8, 4, 6, 0, 3, 7, 4]
three_way_partition(arr, 0, len(arr) - 1)
print(arr)  # 输出: 排序后的数组
```

### 总结

通过这些改进和变化，可以有效地提升快速排序的性能，尤其是在处理不同类型数据和特定情况下。这些方法不仅优化了时间复杂度，还增强了算法的稳定性和适用性。

### 参考文献

1. **GeeksforGeeks - QuickSort on Doubly Linked List**：
   - 提供了详细的快速排序解释和多种实现示例。
   - [链接](https://www.geeksforgeeks.org/quicksort-on-doubly-linked-list/)
   
2. **LeetCode Discuss - Quick Sort**：
   - 提供了快速排序问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/discuss/interview-question/algorithm/317231/quick-sort-algorithm-and-implementation)
