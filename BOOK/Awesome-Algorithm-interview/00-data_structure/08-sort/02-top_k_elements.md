> 堆排序topk



### 堆排序用于查找 Top K 元素

堆排序（Heap Sort）是一种基于堆数据结构的比较排序算法，可以在 $O(n \log k)$ 的时间复杂度内找到数组中最大的 $k$ 个元素。通过维护一个大小为 $k$ 的小顶堆，可以高效地实现这一目标。

#### 算法步骤

1. 创建一个大小为 $k$ 的小顶堆，将前 $k$ 个元素插入堆中。
2. 遍历数组中剩余的元素：
   - 如果当前元素大于堆顶元素，则替换堆顶元素并调整堆。
3. 遍历结束后，堆中的元素即为数组中最大的 $k$ 个元素。

#### 代码实现

```python
import heapq

def top_k_elements(nums, k):
    if k == 0:
        return []
    # 创建一个大小为 k 的小顶堆
    min_heap = nums[:k]
    heapq.heapify(min_heap)
    
    # 遍历数组中剩余的元素
    for num in nums[k:]:
        if num > min_heap[0]:
            heapq.heappushpop(min_heap, num)
    
    # 返回堆中的元素
    return min_heap

# 示例使用
nums = [3, 2, 1, 5, 6, 4]
k = 2
print(top_k_elements(nums, k))  # 输出: [5, 6]
```

### 解释

1. **创建小顶堆**：
   - 使用 `heapq.heapify` 将前 $k$ 个元素构建成一个小顶堆。

2. **遍历数组**：
   - 对于数组中剩余的元素，如果当前元素大于堆顶元素（即堆中的最小元素），则将堆顶元素替换为当前元素，并调整堆使其保持小顶堆性质。

3. **返回结果**：
   - 遍历结束后，堆中的元素即为数组中最大的 $k$ 个元素。

### 时间复杂度和空间复杂度

- **时间复杂度**：$O(n \log k)$，其中 $n$ 是数组的长度。堆的插入和删除操作的时间复杂度为 $O(\log k)$，遍历数组的时间复杂度为 $O(n)$。
- **空间复杂度**：$O(k)$，用于存储堆的额外空间。

### 参考文献

1. **GeeksforGeeks - Kth Largest Element in an Array**：
   - 提供了使用堆查找数组中第 $k$ 大元素的详细解释和实现示例。
   - [GeeksforGeeks链接](https://www.geeksforgeeks.org/kth-largest-element-in-an-array/)
   
2. **LeetCode Problem - Kth Largest Element in an Array**：
   - 提供了查找数组中第 $k$ 大元素问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/kth-largest-element-in-an-array/)