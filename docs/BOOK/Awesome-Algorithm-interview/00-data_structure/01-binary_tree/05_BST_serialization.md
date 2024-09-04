> 如何序列化排序二叉树



### 序列化排序二叉树

排序二叉树（也称二叉搜索树，Binary Search Tree, BST）的序列化是将树结构转换为一种线性表示形式，以便于存储和传输。通常使用前序遍历（preorder traversal）或后序遍历（postorder traversal）来进行序列化，因为这两种遍历方法在给定中序遍历（inorder traversal）的情况下，可以唯一确定一棵BST。

### 前序遍历序列化

前序遍历（Preorder Traversal）是按照根节点 -> 左子树 -> 右子树的顺序遍历二叉树。

#### 序列化代码实现

```python
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def serialize_preorder(root):
    def helper(node):
        if node is None:
            vals.append('#')
            return
        vals.append(str(node.value))
        helper(node.left)
        helper(node.right)
    
    vals = []
    helper(root)
    return ' '.join(vals)

# 示例使用
root = TreeNode(4, TreeNode(2, TreeNode(1), TreeNode(3)), TreeNode(5))
serialized_str = serialize_preorder(root)
print(serialized_str)  # 输出: '4 2 1 # # 3 # # 5 # #'
```

#### 反序列化代码实现

反序列化是将序列化的字符串重新构建为BST。

```python
def deserialize_preorder(data):
    def helper():
        val = next(vals)
        if val == '#':
            return None
        node = TreeNode(int(val))
        node.left = helper()
        node.right = helper()
        return node
    
    vals = iter(data.split())
    return helper()

# 示例使用
deserialized_tree = deserialize_preorder(serialized_str)
print(deserialized_tree.value)  # 输出: 4
```

### 后序遍历序列化

后序遍历（Postorder Traversal）是按照左子树 -> 右子树 -> 根节点的顺序遍历二叉树。

#### 序列化代码实现

```python
def serialize_postorder(root):
    def helper(node):
        if node is None:
            vals.append('#')
            return
        helper(node.left)
        helper(node.right)
        vals.append(str(node.value))
    
    vals = []
    helper(root)
    return ' '.join(vals)

# 示例使用
serialized_str = serialize_postorder(root)
print(serialized_str)  # 输出: '1 # # 3 # # 2 5 # # 4'
```

#### 反序列化代码实现

反序列化是将序列化的字符串重新构建为BST。

```python
def deserialize_postorder(data):
    def helper():
        val = next(vals)
        if val == '#':
            return None
        node = TreeNode(int(val))
        node.right = helper()
        node.left = helper()
        return node
    
    vals = iter(data.split()[::-1])
    return helper()

# 示例使用
deserialized_tree = deserialize_postorder(serialized_str)
print(deserialized_tree.value)  # 输出: 4
```

### 优缺点总结

**前序遍历序列化与反序列化**：
- **优点**：实现简单，便于理解。
- **缺点**：生成的字符串包含大量空节点标记，长度较长。

**后序遍历序列化与反序列化**：
- **优点**：与前序遍历类似，适用于BST的序列化。
- **缺点**：实现稍微复杂，但总体与前序遍历方法类似。

### 参考文献

1. **Introduction to Algorithms, Cormen, Leiserson, Rivest, and Stein**:
   - 提供了各种树的遍历算法的详细描述和伪代码。
   - [书籍链接](https://mitpress.mit.edu/books/introduction-algorithms)

2. **GeeksforGeeks - Serialize and Deserialize a Binary Tree**:
   - 详细介绍了二叉树序列化和反序列化的各种实现方法。
   - [链接](https://www.geeksforgeeks.org/serialize-deserialize-binary-tree/)
