> 两个链表是否有交点



判断两个链表是否有交点（Intersection of Two Linked Lists）是一个经典的问题。可以通过多种方法来解决，其中最常见和高效的方法包括哈希表法和双指针法。

### 方法一：哈希表法

使用哈希表可以简单地解决这个问题。我们遍历第一个链表，将所有节点存入哈希表中。然后遍历第二个链表，检查是否存在节点在哈希表中。如果存在，说明两个链表有交点。

#### 算法步骤

1. 初始化一个空的哈希表 `visited`。
2. 遍历第一个链表，将所有节点存入 `visited`。
3. 遍历第二个链表，检查每个节点是否在 `visited` 中：
   - 如果存在，返回该节点，表示链表有交点。
   - 如果不存在，继续遍历。
4. 如果遍历完第二个链表都没有找到交点，返回 `None`。

#### 代码实现

```python
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

def get_intersection_node(headA, headB):
    visited = set()
    
    current = headA
    while current:
        visited.add(current)
        current = current.next
    
    current = headB
    while current:
        if current in visited:
            return current
        current = current.next
    
    return None

# 示例使用
# 构建两个交叉的链表
# A: 1 -> 2 \
#              6 -> 7
# B: 3 -> 4 -> 5 /
node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(3)
node4 = ListNode(4)
node5 = ListNode(5)
node6 = ListNode(6)
node7 = ListNode(7)

node1.next = node2
node2.next = node6
node6.next = node7

node3.next = node4
node4.next = node5
node5.next = node6

intersection_node = get_intersection_node(node1, node3)
print(intersection_node.value if intersection_node else "No intersection")  # 输出: 6
```

### 方法二：双指针法

双指针法是一种更为高效的方法。我们使用两个指针，分别从两个链表的头节点开始遍历。当指针到达链表末尾时，跳到另一个链表的头节点继续遍历。这样，当两个指针相遇时，指向的就是交点。

#### 算法步骤

1. 初始化两个指针 `pA` 和 `pB`，分别指向链表 `A` 和 `B` 的头节点。
2. 遍历链表：
   - 如果 `pA` 到达链表 `A` 的末尾，则指向链表 `B` 的头节点。
   - 如果 `pB` 到达链表 `B` 的末尾，则指向链表 `A` 的头节点。
   - 如果 `pA` 和 `pB` 相遇，返回该节点，表示链表有交点。
3. 如果两个指针都到达链表末尾仍未相遇，返回 `None`。

#### 代码实现

```python
def get_intersection_node(headA, headB):
    if not headA or not headB:
        return None
    
    pA, pB = headA, headB
    
    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA
    
    return pA

# 示例使用
# 使用之前构建的交叉链表
intersection_node = get_intersection_node(node1, node3)
print(intersection_node.value if intersection_node else "No intersection")  # 输出: 6
```

### 解释

1. **哈希表法**：
   - 使用哈希表存储第一个链表的所有节点。
   - 遍历第二个链表，检查是否存在节点在哈希表中。

2. **双指针法**：
   - 使用两个指针分别遍历两个链表。
   - 当指针到达链表末尾时，跳到另一个链表的头节点继续遍历。
   - 当两个指针相遇时，即为交点。

### 优缺点

**哈希表法**：
- **优点**：实现简单，逻辑清晰。
- **缺点**：需要额外的哈希表存储空间，空间复杂度为 $O(n)$。

**双指针法**：
- **优点**：时间复杂度为 $O(n + m)$，空间复杂度为 $O(1)$。
- **缺点**：需要两个链表的长度差不大时效果更好。

### 参考文献

1. **GeeksforGeeks - Intersection Point in Y Shapped Linked Lists**：
   - 提供了详细的交点查找方法和代码实现。
   - [链接](https://www.geeksforgeeks.org/write-a-function-to-get-the-intersection-point-of-two-linked-lists/)

2. **LeetCode Problem - Intersection of Two Linked Lists**：
   - 提供了链表交点问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/intersection-of-two-linked-lists/)