> 判断二叉树是否相同？判断一棵树是不是另一棵树的子树？


### 判断二叉树是否相同

判断两棵二叉树是否相同，即判断它们的结构和节点值是否完全一致。可以使用递归的方法来实现。

#### 递归方法

**算法步骤**：
1. 如果两棵树都为空，则它们相同。
2. 如果一棵树为空，另一棵树不为空，则它们不同。
3. 如果两棵树的根节点值不同，则它们不同。
4. 递归判断左子树和右子树是否相同。

**实现代码**：

```python
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def is_same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    if p.value != q.value:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)

# 示例使用
p = TreeNode(1, TreeNode(2), TreeNode(3))
q = TreeNode(1, TreeNode(2), TreeNode(3))
print(is_same_tree(p, q))  # 输出: True
```

### 判断一棵树是否是另一棵树的子树

判断二叉树 `t` 是否是二叉树 `s` 的子树，可以使用递归和辅助函数的方法。

#### 递归方法

**算法步骤**：
1. 如果 `t` 为空，则 `t` 是 `s` 的子树。
2. 如果 `s` 为空，则 `t` 不是 `s` 的子树。
3. 判断 `s` 和 `t` 是否相同。
4. 递归判断 `t` 是否是 `s` 的左子树的子树。
5. 递归判断 `t` 是否是 `s` 的右子树的子树。

**实现代码**：

```python
def is_subtree(s, t):
    if not t:
        return True
    if not s:
        return False
    if is_same_tree(s, t):
        return True
    return is_subtree(s.left, t) or is_subtree(s.right, t)

# 示例使用
s = TreeNode(3, TreeNode(4, TreeNode(1), TreeNode(2)), TreeNode(5))
t = TreeNode(4, TreeNode(1), TreeNode(2))
print(is_subtree(s, t))  # 输出: True
```

### 参考文献

1. **GeeksforGeeks - Check if two trees are identical**：
   - 详细介绍了判断两棵二叉树是否相同的方法和实现。
   - [链接](https://www.geeksforgeeks.org/write-c-code-to-determine-if-two-trees-are-identical/)

2. **LeetCode Problem - Same Tree**：
   - 提供了关于判断两棵二叉树是否相同的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/same-tree/)

3. **GeeksforGeeks - Check if a binary tree is subtree of another binary tree**：
   - 详细介绍了判断一棵树是否是另一棵树的子树的方法和实现。
   - [链接](https://www.geeksforgeeks.org/check-if-a-binary-tree-is-subtree-of-another-binary-tree/)