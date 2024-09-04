> 前中后的非递归？


### 非递归遍历二叉树

二叉树的遍历是指按照一定的顺序访问二叉树的每一个节点，通常有前序遍历（Pre-order Traversal）、中序遍历（In-order Traversal）和后序遍历（Post-order Traversal）三种方式。非递归遍历是指不使用递归的方法来实现这些遍历。下面详细介绍如何使用栈来实现这三种遍历方式。

#### 1. 前序遍历（Pre-order Traversal）

前序遍历的顺序是：根节点 -> 左子树 -> 右子树。

**算法**：
1. 初始化一个空栈，将根节点压入栈中。
2. 当栈不为空时：
   - 弹出栈顶节点并访问它。
   - 将该节点的右子节点压入栈（如果存在）。
   - 将该节点的左子节点压入栈（如果存在）。

**实现**：
```python
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def pre_order_traversal(root):
    if not root:
        return []
    stack, result = [root], []
    while stack:
        node = stack.pop()
        if node:
            result.append(node.value)
            stack.append(node.right)
            stack.append(node.left)
    return result

# 示例使用
root = TreeNode(1, TreeNode(2), TreeNode(3))
print(pre_order_traversal(root))  # 输出: [1, 2, 3]
```

#### 2. 中序遍历（In-order Traversal）

中序遍历的顺序是：左子树 -> 根节点 -> 右子树。

**算法**：
1. 初始化一个空栈和一个指向当前节点的指针 `curr`，初始为根节点。
2. 当 `curr` 不为空或栈不为空时：
   - 当 `curr` 不为空时，将 `curr` 压入栈并移动 `curr` 指向左子节点。
   - 当 `curr` 为空时，弹出栈顶节点，将该节点值添加到结果中，并将 `curr` 移动到该节点的右子节点。

**实现**：
```python
def in_order_traversal(root):
    stack, result, curr = [], [], root
    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.value)
        curr = curr.right
    return result

# 示例使用
print(in_order_traversal(root))  # 输出: [2, 1, 3]
```

#### 3. 后序遍历（Post-order Traversal）

后序遍历的顺序是：左子树 -> 右子树 -> 根节点。

**算法**：
1. 初始化两个空栈 `stack1` 和 `stack2`，将根节点压入 `stack1`。
2. 当 `stack1` 不为空时：
   - 弹出 `stack1` 栈顶节点并压入 `stack2`。
   - 将该节点的左子节点压入 `stack1`（如果存在）。
   - 将该节点的右子节点压入 `stack1`（如果存在）。
3. 将 `stack2` 中的所有节点值依次弹出，得到后序遍历结果。

**实现**：
```python
def post_order_traversal(root):
    if not root:
        return []
    stack1, stack2, result = [root], [], []
    while stack1:
        node = stack1.pop()
        stack2.append(node)
        if node.left:
            stack1.append(node.left)
        if node.right:
            stack1.append(node.right)
    while stack2:
        result.append(stack2.pop().value)
    return result

# 示例使用
print(post_order_traversal(root))  # 输出: [2, 3, 1]
```

### 总结

- **前序遍历**：使用栈模拟递归，通过先压右子节点再压左子节点的方式实现。
- **中序遍历**：利用栈和指针，实现先到达最左端，然后回溯并访问右子节点的逻辑。
- **后序遍历**：使用两个栈，一个栈用来存储访问顺序，另一个栈用来逆序存储最终的结果。
