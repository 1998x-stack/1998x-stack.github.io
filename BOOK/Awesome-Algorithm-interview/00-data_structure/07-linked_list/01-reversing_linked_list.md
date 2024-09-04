> 翻转链表


### 翻转链表（Reversing a Linked List）

翻转链表是一个常见的链表操作，可以使用迭代或递归方法来实现。下面是详细的解释和代码实现。

### 方法一：迭代法

通过迭代方法翻转链表，我们需要三个指针来跟踪链表的当前节点、前一个节点和下一个节点。

#### 算法步骤

1. 初始化指针 `prev` 为 `None`，`curr` 为链表的头节点 `head`。
2. 遍历链表，直到 `curr` 为空：
   - 保存当前节点的下一个节点 `next_temp`。
   - 将当前节点的 `next` 指向前一个节点 `prev`。
   - 将前一个节点 `prev` 移动到当前节点 `curr`。
   - 将当前节点 `curr` 移动到下一个节点 `next_temp`。
3. 返回新的头节点 `prev`。

#### 代码实现

```python
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

def reverse_list(head):
    prev = None
    curr = head

    while curr is not None:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp

    return prev

# 示例使用
def print_list(node):
    while node:
        print(node.value, end=" -> ")
        node = node.next
    print("None")

# 构建链表 1 -> 2 -> 3 -> 4 -> 5 -> None
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)
head.next.next.next.next = ListNode(5)

print("原链表:")
print_list(head)

reversed_head = reverse_list(head)
print("翻转后的链表:")
print_list(reversed_head)
```

### 方法二：递归法

通过递归方法翻转链表，主要思想是通过递归处理子链表，最终将每个节点的 `next` 指向前一个节点。

#### 算法步骤

1. 递归终止条件：如果链表为空或只有一个节点，直接返回头节点。
2. 递归翻转子链表，直到处理到最后一个节点。
3. 翻转当前节点与其下一个节点的指向关系。
4. 返回新的头节点。

#### 代码实现

```python
def reverse_list_recursive(head):
    if head is None or head.next is None:
        return head
    
    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    
    return new_head

# 示例使用
print("原链表:")
print_list(head)

reversed_head_recursive = reverse_list_recursive(head)
print("翻转后的链表（递归）:")
print_list(reversed_head_recursive)
```

### 解释

1. **迭代法**：
   - 使用三个指针 `prev`、`curr` 和 `next_temp`。
   - 遍历链表并逐步翻转指向关系，直到所有节点都被处理。

2. **递归法**：
   - 递归处理子链表，直到链表尾部。
   - 翻转当前节点与其下一个节点的指向关系。
   - 返回新的头节点。

### 优缺点

**迭代法**：
- **优点**：实现简单，时间复杂度为 $O(n)$，空间复杂度为 $O(1)$。
- **缺点**：需要显式维护多个指针，代码较为繁琐。

**递归法**：
- **优点**：代码简洁，逻辑清晰。
- **缺点**：递归调用消耗栈空间，空间复杂度为 $O(n)$。

### 参考文献

1. **GeeksforGeeks - Reverse a linked list**：
   - 提供了迭代法和递归法翻转链表的详细解释和代码示例。
   - [链接](https://www.geeksforgeeks.org/reverse-a-linked-list/)
   
2. **LeetCode Problem - Reverse Linked List**：
   - 提供了翻转链表问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/reverse-linked-list/)