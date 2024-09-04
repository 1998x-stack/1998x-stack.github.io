> 逆序打印链表


逆序打印链表是一个常见的编程问题，可以通过多种方法来实现。以下是几种常见的解决方案，包括递归法和使用栈的方法。

### 方法一：递归法

递归方法利用系统调用栈的特性，可以方便地实现逆序打印链表。递归调用将链表节点压栈，递归返回时依次打印节点。

#### 算法步骤

1. 定义递归函数 `print_reverse`，参数为当前节点。
2. 在递归函数中，先递归调用 `print_reverse` 打印下一个节点。
3. 打印当前节点的值。

#### 代码实现

```python
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

def print_reverse(node):
    if node is None:
        return
    print_reverse(node.next)
    print(node.value)

# 示例使用
# 构建链表 1 -> 2 -> 3 -> 4 -> 5 -> None
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))

print("逆序打印链表（递归法）:")
print_reverse(head)
```

### 方法二：使用栈

使用栈可以模拟递归的行为，栈的特性是后进先出（LIFO），可以实现逆序打印链表。

#### 算法步骤

1. 初始化一个空栈 `stack`。
2. 遍历链表，将每个节点压入栈中。
3. 遍历栈，依次弹出节点并打印其值。

#### 代码实现

```python
def print_reverse_stack(node):
    stack = []
    while node:
        stack.append(node)
        node = node.next
    while stack:
        node = stack.pop()
        print(node.value)

# 示例使用
print("逆序打印链表（使用栈）:")
print_reverse_stack(head)
```

### 解释

1. **递归法**：
   - 递归函数 `print_reverse` 先递归调用自己打印下一个节点，递归返回时再打印当前节点的值。
   - 利用系统调用栈，实现逆序打印。

2. **使用栈**：
   - 遍历链表，将节点依次压入栈中。
   - 遍历栈，依次弹出节点并打印其值，实现逆序打印。

### 优缺点

**递归法**：
- **优点**：代码简洁，逻辑清晰。
- **缺点**：对于链表长度较大时，递归深度可能导致栈溢出（Stack Overflow）。

**使用栈**：
- **优点**：避免了递归深度限制，适用于较长链表。
- **缺点**：需要额外的栈空间，空间复杂度为 $O(n)$。

### 参考文献

1. **GeeksforGeeks - Reverse a linked list**：
   - 提供了链表逆序打印的详细解释和代码实现。
   - [链接](https://www.geeksforgeeks.org/reverse-a-linked-list/)
   
2. **LeetCode Discuss - Print a linked list in reverse order**：
   - 提供了链表逆序打印问题的详细描述和多种解决方案。
   - [LeetCode链接](https://leetcode.com/discuss/interview-question/356150/print-a-linked-list-in-reverse-order)