> 二叉树的路径和为定值的路径



在二叉树中，寻找所有路径和等于一个给定值的路径是一个经典问题。路径可以从任意节点开始，到任意节点结束，但路径必须是从父节点到子节点的方向（即不允许反向）。这里提供一个解决该问题的算法和其Python实现。

### 算法步骤

1. 定义一个递归函数 `find_paths`，该函数的参数包括当前节点、目标和 `target_sum`、当前路径 `current_path` 和保存结果的列表 `all_paths`。
2. 在函数 `find_paths` 中：
   - 如果当前节点为空，直接返回。
   - 将当前节点的值添加到 `current_path`。
   - 初始化 `path_sum` 为0，并从 `current_path` 的最后一个元素向前遍历，计算所有以当前节点结尾的子路径的和。如果某个子路径和等于 `target_sum`，将这个子路径加入 `all_paths`。
   - 递归调用 `find_paths`，分别对当前节点的左子节点和右子节点进行相同操作。
   - 递归返回前，从 `current_path` 中移除当前节点的值（回溯）。
3. 调用函数 `find_paths` 开始计算。

### 代码实现

```python
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def find_paths(root, target_sum):
    def find_paths_recursive(current_node, target_sum, current_path, all_paths):
        if not current_node:
            return

        # 添加当前节点到路径
        current_path.append(current_node.value)

        # 检查以当前节点为结尾的所有路径是否等于target_sum
        path_sum = 0
        for i in range(len(current_path) - 1, -1, -1):
            path_sum += current_path[i]
            if path_sum == target_sum:
                all_paths.append(list(current_path[i:]))

        # 递归处理左子树和右子树
        find_paths_recursive(current_node.left, target_sum, current_path, all_paths)
        find_paths_recursive(current_node.right, target_sum, current_path, all_paths)

        # 回溯：移除当前节点
        current_path.pop()

    all_paths = []
    find_paths_recursive(root, target_sum, [], all_paths)
    return all_paths

# 示例使用
root = TreeNode(5)
root.left = TreeNode(4)
root.right = TreeNode(8)
root.left.left = TreeNode(11)
root.left.left.left = TreeNode(7)
root.left.left.right = TreeNode(2)
root.right.left = TreeNode(13)
root.right.right = TreeNode(4)
root.right.right.left = TreeNode(5)
root.right.right.right = TreeNode(1)

target_sum = 22
paths = find_paths(root, target_sum)
for path in paths:
    print(path)  # 输出: [[5, 4, 11, 2], [5, 8, 4, 5]]
```

### 复杂度分析

- **时间复杂度**：O(N^2)，其中 N 是节点的数量。在最坏情况下，对于每个节点，路径和的检查需要O(N)的时间。
- **空间复杂度**：O(N)，用于存储当前路径和递归调用栈。

### 参考文献

1. **LeetCode Problem - Path Sum II**:
   - 提供了寻找路径和等于给定值问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/path-sum-ii/)

2. **GeeksforGeeks - Print all paths with sum k in a binary tree**:
   - 详细介绍了在二叉树中找到所有路径和等于给定值的各种实现方法。
   - [GeeksforGeeks链接](https://www.geeksforgeeks.org/print-k-sum-paths-binary-tree/)