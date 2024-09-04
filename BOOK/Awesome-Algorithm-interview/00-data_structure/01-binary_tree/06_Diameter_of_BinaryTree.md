> 二叉树的直径


二叉树的直径（Diameter of a Binary Tree）是指二叉树中任意两个节点之间路径上的节点数的最大值。这个路径可能通过根节点，也可能不通过根节点。计算二叉树的直径的常用方法是通过递归，计算每个节点的左右子树的深度，并更新直径的最大值。

### 算法步骤

1. 定义一个全局变量 `max_diameter`，用来存储当前计算出的最大直径。
2. 定义一个递归函数 `depth(node)`，该函数返回以 `node` 为根节点的子树的深度，并在计算过程中更新 `max_diameter`。
3. 在 `depth(node)` 函数中：
   - 如果节点 `node` 为空，返回深度 0。
   - 递归计算左子树的深度 `left_depth` 和右子树的深度 `right_depth`。
   - 更新 `max_diameter` 为 `left_depth + right_depth` 和当前 `max_diameter` 的较大值。
   - 返回当前节点的深度，即 `max(left_depth, right_depth) + 1`。
4. 调用 `depth(root)` 函数开始计算。

### 代码实现

以下是 Python 实现计算二叉树直径的代码：

```python
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def diameter_of_binary_tree(root):
    def depth(node):
        nonlocal max_diameter
        if not node:
            return 0
        left_depth = depth(node.left)
        right_depth = depth(node.right)
        max_diameter = max(max_diameter, left_depth + right_depth)
        return max(left_depth, right_depth) + 1
    
    max_diameter = 0
    depth(root)
    return max_diameter

# 示例使用
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

print(diameter_of_binary_tree(root))  # 输出: 3
```

### 解释

- **`depth(node)`**：递归函数，返回以 `node` 为根节点的子树的深度。
  - 如果 `node` 为空，返回深度 0。
  - 计算左子树和右子树的深度。
  - 更新 `max_diameter` 为当前节点的左右子树深度之和与 `max_diameter` 的较大值。
  - 返回当前节点的深度，即 `max(left_depth, right_depth) + 1`。

- **`diameter_of_binary_tree(root)`**：主函数，初始化 `max_diameter` 并调用 `depth(root)` 开始计算。

### 复杂度分析

- **时间复杂度**：O(N)，其中 N 是二叉树中的节点数。每个节点只被访问一次。
- **空间复杂度**：O(H)，其中 H 是二叉树的高度。递归调用栈的最大深度为树的高度。

### 参考文献

1. **LeetCode Problem - Diameter of Binary Tree**:
   - 提供了二叉树直径问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/diameter-of-binary-tree/)

2. **GeeksforGeeks - Diameter of a Binary Tree**:
   - 详细介绍了如何计算二叉树的直径，并提供了多种实现方法。
   - [GeeksforGeeks链接](https://www.geeksforgeeks.org/diameter-of-a-binary-tree/)