> 排序二叉树转双向链表



将二叉搜索树（BST）转换为双向链表是一个经典的问题，可以通过中序遍历的方法实现，因为中序遍历会按照升序访问BST中的节点。我们希望转换后的双向链表按照节点值的顺序排列，并且链表的节点应该链接成双向的，即每个节点都有指向前驱和后继的指针。

### 转换步骤

1. **定义辅助函数**：定义一个辅助函数 `convert(node)`，用于递归地将BST转换为双向链表。
2. **递归过程**：
   - 如果当前节点为空，返回。
   - 递归地将左子树转换为双向链表。
   - 将当前节点链接到双向链表的尾部。
   - 递归地将右子树转换为双向链表。
3. **维护双向链表的尾节点**：在递归过程中，维护一个指向链表尾节点的指针，便于将当前节点链接到链表中。

### 代码实现

```python
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

class DoublyListNode:
    def __init__(self, value=0, prev=None, next=None):
        self.value = value
        self.prev = prev
        self.next = next

def bst_to_doubly_linked_list(root):
    if not root:
        return None
    
    # Dummy node to act as the predecessor of the list's head
    dummy = DoublyListNode(0)
    prev = dummy  # This will eventually be the tail of the list
    
    def convert(node):
        nonlocal prev
        if not node:
            return
        
        # Recursively convert the left subtree
        convert(node.left)
        
        # Convert the current node
        current = DoublyListNode(node.value)
        prev.next = current
        current.prev = prev
        prev = current
        
        # Recursively convert the right subtree
        convert(node.right)
    
    convert(root)
    
    # Head of the doubly linked list
    head = dummy.next
    if head:
        head.prev = None
    return head

# 示例使用
root = TreeNode(4, TreeNode(2, TreeNode(1), TreeNode(3)), TreeNode(5))
head = bst_to_doubly_linked_list(root)

# 打印双向链表
current = head
while current:
    print(current.value, end=" <-> " if current.next else "")
    current = current.next
```

### 解释

1. **TreeNode 类**：定义二叉树节点类，每个节点包含一个值和指向左、右子节点的指针。
2. **DoublyListNode 类**：定义双向链表节点类，每个节点包含一个值和指向前驱、后继节点的指针。
3. **bst_to_doubly_linked_list 函数**：
   - 使用一个虚拟头节点 `dummy` 来简化链表的操作。
   - 递归函数 `convert(node)` 用于中序遍历BST，并构建双向链表。
   - 维护一个指向链表尾节点的指针 `prev`，以便将当前节点链接到链表中。
4. **递归转换**：
   - 先递归转换左子树，再处理当前节点，最后递归转换右子树。
   - 当前节点被转换为双向链表节点，并链接到链表的尾部。

### 参考文献

1. **GeeksforGeeks - Convert a given Binary Tree to Doubly Linked List**：
   - 提供了关于将二叉树转换为双向链表的详细解释和多种实现方法。
   - [链接](https://www.geeksforgeeks.org/convert-given-binary-tree-doubly-linked-list-set-3/)

2. **LeetCode Discuss - Convert Binary Search Tree to Sorted Doubly Linked List**：
   - 提供了详细的讨论和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/)

