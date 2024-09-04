> 前中，后中遍历结果恢复二叉树

要恢复二叉树，可以使用前序遍历（preorder）和中序遍历（inorder）的结果，或后序遍历（postorder）和中序遍历（inorder）的结果。以下详细介绍如何通过这两种方法来恢复二叉树。

### 方法一：使用前序遍历和中序遍历结果恢复二叉树

前序遍历结果：根 -> 左子树 -> 右子树  
中序遍历结果：左子树 -> 根 -> 右子树

**算法**：
1. 前序遍历的第一个元素是根节点。
2. 在中序遍历中找到根节点的位置，以此位置将中序遍历划分为左子树和右子树。
3. 递归地对前序遍历和中序遍历的左右子树部分重复上述步骤。

**实现代码**：
```python
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def build_tree_pre_in(preorder, inorder):
    if not preorder or not inorder:
        return None

    # 根节点是前序遍历的第一个元素
    root_value = preorder[0]
    root = TreeNode(root_value)

    # 在中序遍历中找到根节点的位置
    root_index = inorder.index(root_value)

    # 划分左右子树
    left_inorder = inorder[:root_index]
    right_inorder = inorder[root_index + 1:]
    left_preorder = preorder[1:1 + len(left_inorder)]
    right_preorder = preorder[1 + len(left_inorder):]

    # 递归构建左右子树
    root.left = build_tree_pre_in(left_preorder, left_inorder)
    root.right = build_tree_pre_in(right_preorder, right_inorder)

    return root

# 示例使用
preorder = [1, 2, 4, 5, 3, 6, 7]
inorder = [4, 2, 5, 1, 6, 3, 7]
root = build_tree_pre_in(preorder, inorder)
```

### 方法二：使用后序遍历和中序遍历结果恢复二叉树

后序遍历结果：左子树 -> 右子树 -> 根  
中序遍历结果：左子树 -> 根 -> 右子树

**算法**：
1. 后序遍历的最后一个元素是根节点。
2. 在中序遍历中找到根节点的位置，以此位置将中序遍历划分为左子树和右子树。
3. 递归地对后序遍历和中序遍历的左右子树部分重复上述步骤。

**实现代码**：
```python
def build_tree_post_in(postorder, inorder):
    if not postorder or not inorder:
        return None

    # 根节点是后序遍历的最后一个元素
    root_value = postorder[-1]
    root = TreeNode(root_value)

    # 在中序遍历中找到根节点的位置
    root_index = inorder.index(root_value)

    # 划分左右子树
    left_inorder = inorder[:root_index]
    right_inorder = inorder[root_index + 1:]
    left_postorder = postorder[:len(left_inorder)]
    right_postorder = postorder[len(left_inorder):-1]

    # 递归构建左右子树
    root.left = build_tree_post_in(left_postorder, left_inorder)
    root.right = build_tree_post_in(right_postorder, right_inorder)

    return root

# 示例使用
postorder = [4, 5, 2, 6, 7, 3, 1]
inorder = [4, 2, 5, 1, 6, 3, 7]
root = build_tree_post_in(postorder, inorder)
```

### 总结

- **前序和中序遍历结果恢复二叉树**：前序遍历的第一个元素是根节点，在中序遍历中找到根节点位置，划分左右子树，递归构建子树。
- **后序和中序遍历结果恢复二叉树**：后序遍历的最后一个元素是根节点，在中序遍历中找到根节点位置，划分左右子树，递归构建子树。