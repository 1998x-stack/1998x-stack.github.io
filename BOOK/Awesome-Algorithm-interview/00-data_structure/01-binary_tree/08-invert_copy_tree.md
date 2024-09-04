> 翻转、复制二叉树


### 翻转二叉树

翻转二叉树是指交换每个节点的左子树和右子树，从而得到镜像二叉树。这可以通过递归或迭代的方法实现。

#### 递归方法

**算法步骤**：
1. 如果当前节点为空，返回。
2. 交换当前节点的左子节点和右子节点。
3. 递归翻转左子树。
4. 递归翻转右子树。

**实现代码**：

```python
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def invert_tree(root):
    if not root:
        return None
    
    # 交换左子树和右子树
    root.left, root.right = root.right, root.left
    
    # 递归翻转左子树和右子树
    invert_tree(root.left)
    invert_tree(root.right)
    
    return root

# 示例使用
root = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
inverted_root = invert_tree(root)
```

#### 迭代方法

**算法步骤**：
1. 初始化一个队列，将根节点加入队列。
2. 当队列不为空时：
   - 取出队列中的节点，交换其左子节点和右子节点。
   - 将其左子节点和右子节点（如果存在）加入队列。

**实现代码**：

```python
from collections import deque

def invert_tree_iterative(root):
    if not root:
        return None
    
    queue = deque([root])
    while queue:
        node = queue.popleft()
        node.left, node.right = node.right, node.left
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return root

# 示例使用
inverted_root_iter = invert_tree_iterative(root)
```

### 复制二叉树

复制二叉树是指创建一个与原二叉树结构和节点值完全相同的新的二叉树。这可以通过递归的方法实现。

#### 递归方法

**算法步骤**：
1. 如果当前节点为空，返回空节点。
2. 创建一个新的节点，其值等于当前节点的值。
3. 递归复制左子树，并将其赋值给新节点的左子树。
4. 递归复制右子树，并将其赋值给新节点的右子树。

**实现代码**：

```python
def copy_tree(root):
    if not root:
        return None
    
    new_root = TreeNode(root.value)
    new_root.left = copy_tree(root.left)
    new_root.right = copy_tree(root.right)
    
    return new_root

# 示例使用
copied_root = copy_tree(root)
```

### 参考文献

1. **GeeksforGeeks - Invert/Reverse a Binary Tree**:
   - 提供了详细的翻转二叉树的实现和说明。
   - [链接](https://www.geeksforgeeks.org/invert-binary-tree/)

2. **LeetCode Problem - Invert Binary Tree**:
   - 提供了关于翻转二叉树的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/invert-binary-tree/)

3. **GeeksforGeeks - Clone a Binary Tree**:
   - 详细介绍了如何复制二叉树，并提供了实现方法。
   - [链接](https://www.geeksforgeeks.org/clone-binary-tree/)