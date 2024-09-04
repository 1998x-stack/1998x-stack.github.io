> 合并链表


合并链表有多种形式，最常见的有以下几种：

1. **合并两个有序链表**：将两个升序链表合并为一个新的升序链表。
2. **合并多个有序链表**：将多个升序链表合并为一个新的升序链表。
3. **按顺序合并链表**：将两个链表的节点交替合并。

### 合并两个有序链表

合并两个有序链表的基本思想是同时遍历两个链表，比较它们的节点值，将较小的节点添加到新链表中，直到遍历完其中一个链表，然后将另一个链表的剩余部分添加到新链表中。

#### 算法步骤

1. 创建一个虚拟头节点 `dummy`，用于简化操作，并初始化一个指针 `current` 指向虚拟头节点。
2. 同时遍历两个链表，比较当前节点值，将较小的节点连接到 `current` 节点后面，并将指针移动到下一个节点。
3. 当一个链表遍历完，将另一个链表的剩余部分连接到 `current` 节点后面。
4. 返回虚拟头节点的下一个节点作为新链表的头节点。

#### 代码实现

```python
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

def merge_two_lists(l1, l2):
    dummy = ListNode()
    current = dummy

    while l1 and l2:
        if l1.value < l2.value:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next

    current.next = l1 if l1 else l2

    return dummy.next

# 示例使用
# 构建两个有序链表
# l1: 1 -> 3 -> 5
# l2: 2 -> 4 -> 6
l1 = ListNode(1, ListNode(3, ListNode(5)))
l2 = ListNode(2, ListNode(4, ListNode(6)))

merged_list = merge_two_lists(l1, l2)

# 打印合并后的链表
def print_list(node):
    while node:
        print(node.value, end=" -> ")
        node = node.next
    print("None")

print_list(merged_list)
```

### 合并多个有序链表

对于合并多个有序链表，可以使用最小堆（优先队列）来高效地进行合并。通过将每个链表的头节点插入最小堆，每次从堆中取出最小节点并将其下一个节点插入堆中。

#### 代码实现

```python
import heapq

def merge_k_lists(lists):
    dummy = ListNode()
    current = dummy
    heap = []

    # 初始化堆
    for l in lists:
        if l:
            heapq.heappush(heap, (l.value, l))
    
    while heap:
        value, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        if node.next:
            heapq.heappush(heap, (node.next.value, node.next))
    
    return dummy.next

# 示例使用
# 构建多个有序链表
# l1: 1 -> 4 -> 7
# l2: 2 -> 5 -> 8
# l3: 3 -> 6 -> 9
l1 = ListNode(1, ListNode(4, ListNode(7)))
l2 = ListNode(2, ListNode(5, ListNode(8)))
l3 = ListNode(3, ListNode(6, ListNode(9)))

merged_list = merge_k_lists([l1, l2, l3])
print_list(merged_list)
```

### 按顺序合并链表

按顺序合并两个链表的基本思想是交替连接两个链表的节点，直到其中一个链表遍历完，然后将另一个链表的剩余部分连接到新链表中。

#### 代码实现

```python
def merge_alternate(l1, l2):
    dummy = ListNode()
    current = dummy
    toggle = True

    while l1 and l2:
        if toggle:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
        toggle = not toggle

    current.next = l1 if l1 else l2

    return dummy.next

# 示例使用
# 构建两个链表
# l1: 1 -> 3 -> 5
# l2: 2 -> 4 -> 6
l1 = ListNode(1, ListNode(3, ListNode(5)))
l2 = ListNode(2, ListNode(4, ListNode(6)))

merged_list = merge_alternate(l1, l2)
print_list(merged_list)
```

### 参考文献

1. **LeetCode Problem - Merge Two Sorted Lists**：
   - 提供了合并两个有序链表问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/merge-two-sorted-lists/)
   
2. **LeetCode Problem - Merge k Sorted Lists**：
   - 提供了合并多个有序链表问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/merge-k-sorted-lists/)
   
3. **GeeksforGeeks - Merge two sorted linked lists**：
   - 提供了合并两个有序链表的详细解释和代码实现。
   - [链接](https://www.geeksforgeeks.org/merge-two-sorted-linked-lists/)