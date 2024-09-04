> 数组中的topk



在数组中查找前 $ k $ 个最大元素（Top K 问题）是一类常见的算法问题，有多种解决方法。以下是几种常见的方法及其实现：

### 1. 基于堆的方法

利用小顶堆（Min-Heap）来维护当前最大的 $ k $ 个元素。这样可以在 $ O(n \log k) $ 的时间复杂度内找到前 $ k $ 个最大元素。

**步骤**：
1. 初始化一个大小为 $ k $ 的小顶堆。
2. 遍历数组，对于每个元素：
   - 如果堆的大小小于 $ k $，则将元素添加到堆中。
   - 否则，比较当前元素与堆顶元素，如果当前元素较大，则替换堆顶元素并调整堆。

**代码实现**：
```python
import heapq

def top_k_elements(nums, k):
    if k == 0:
        return []
    # 使用小顶堆
    min_heap = []
    for num in nums:
        if len(min_heap) < k:
            heapq.heappush(min_heap, num)
        else:
            if num > min_heap[0]:
                heapq.heappushpop(min_heap, num)
    return min_heap

# 示例使用
nums = [3, 2, 1, 5, 6, 4]
k = 2
print(top_k_elements(nums, k))  # 输出: [5, 6]
```

### 2. 基于快速选择（Quick Select）的方法

快速选择是一种类似于快速排序（Quick Sort）的分治算法。它可以在平均 $ O(n) $ 的时间复杂度内解决Top K问题。

**步骤**：
1. 选择一个枢轴（pivot），将数组划分为两部分，使得一部分小于枢轴，另一部分大于枢轴。
2. 判断枢轴的位置与 $ k $ 的关系：
   - 如果枢轴的位置恰好为 $ k $，则枢轴及其右边的元素即为前 $ k $ 个最大元素。
   - 如果枢轴的位置小于 $ k $，则在右子数组中继续查找。
   - 否则，在左子数组中继续查找。

**代码实现**：
```python
import random

def quick_select(nums, k):
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        # 移动枢轴到末尾
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot_value:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        nums[right], nums[store_index] = nums[store_index], nums[right]
        return store_index
    
    def select(left, right, k_smallest):
        if left == right:
            return nums[left]
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        if k_smallest == pivot_index:
            return nums[k_smallest]
        elif k_smallest < pivot_index:
            return select(left, pivot_index - 1, k_smallest)
        else:
            return select(pivot_index + 1, right, k_smallest)
    
    return select(0, len(nums) - 1, len(nums) - k)

# 示例使用
nums = [3, 2, 1, 5, 6, 4]
k = 2
print(quick_select(nums, k))  # 输出: 5 (注意这是第 k 大元素，不是前 k 大的列表)
```

### 3. 基于排序的方法

直接对数组进行排序，然后取前 $ k $ 个元素。虽然这种方法时间复杂度为 $ O(n \log n) $，但在某些情况下简单易实现。

**代码实现**：
```python
def top_k_elements_sort(nums, k):
    return sorted(nums, reverse=True)[:k]

# 示例使用
nums = [3, 2, 1, 5, 6, 4]
k = 2
print(top_k_elements_sort(nums, k))  # 输出: [6, 5]
```

### 比较与总结

- **基于堆的方法**：时间复杂度为 $ O(n \log k) $，适合处理大规模数据。
- **快速选择**：平均时间复杂度为 $ O(n) $，在需要频繁查询的情况下效率较高。
- **排序方法**：时间复杂度为 $ O(n \log n) $，实现简单但不适合大规模数据。