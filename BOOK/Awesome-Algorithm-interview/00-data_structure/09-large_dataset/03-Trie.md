> trie树

Trie树（也称为前缀树或字典树）是一种树形数据结构，用于存储键集合，其中键通常是字符串。Trie树特别适合处理字符串查找操作，如前缀匹配。以下是关于Trie树的详细展开和实现步骤：

### Trie树的特点
1. **节点结构**：
   - 每个节点代表一个字符。
   - 根节点为空字符。
   - 每个节点包含一个子节点字典，子节点字典的键是字符，值是Trie树的子节点。
   - 每个节点有一个布尔标志，表示是否是一个单词的结尾。

2. **操作**：
   - **插入**：将一个单词插入到Trie树中。
   - **查找**：查找一个单词是否在Trie树中。
   - **前缀查找**：查找所有以某个前缀开头的单词。

### Trie树的实现步骤

#### 步骤一：定义Trie节点类
```python
class TrieNode:
    def __init__(self):
        self.children = {}  # 孩子节点字典
        self.is_end_of_word = False  # 是否是单词结尾
```

#### 步骤二：定义Trie树类
```python
class Trie:
    def __init__(self):
        self.root = TrieNode()  # 根节点

    def insert(self, word: str) -> None:
        """插入单词"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """查找单词"""
        node = self._find_node(word)
        return node is not None and node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """查找前缀"""
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> TrieNode:
        """辅助函数：查找前缀节点"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
```

### Trie树操作示例
```python
# 创建Trie树
trie = Trie()
# 插入单词
trie.insert("apple")
trie.insert("app")
trie.insert("application")

# 查找单词
print(trie.search("apple"))  # 输出: True
print(trie.search("app"))    # 输出: True
print(trie.search("appl"))   # 输出: False

# 查找前缀
print(trie.starts_with("app"))  # 输出: True
print(trie.starts_with("apl"))  # 输出: False
```

### 代码详细解释
1. **TrieNode类**：
   - `self.children`：字典，用于存储子节点，键是字符，值是TrieNode对象。
   - `self.is_end_of_word`：布尔变量，表示该节点是否是一个单词的结尾。

2. **Trie类**：
   - `self.root`：根节点，初始化为空的TrieNode。
   - `insert`方法：逐字符插入单词，如果字符不存在，则创建新节点。最后一个字符的节点标记为单词结尾。
   - `search`方法：查找单词是否存在，利用`_find_node`方法找到最后一个字符的节点，并检查`is_end_of_word`标志。
   - `starts_with`方法：检查前缀是否存在，只需利用`_find_node`方法查找到前缀的最后一个字符的节点即可。
   - `_find_node`方法：辅助方法，逐字符查找前缀，返回最后一个字符的节点。

### 进一步扩展
可以在Trie树的基础上实现更多功能，如：
- **删除单词**：实现删除功能，需要处理子节点的清理。
- **统计前缀数量**：统计以某个前缀开头的单词数量。
- **自动补全**：实现自动补全功能，返回所有以某个前缀开头的单词列表。

#### 删除单词
```python
def delete(self, word: str) -> None:
    """删除单词"""
    def _delete(node: TrieNode, word: str, depth: int) -> bool:
        if depth == len(word):
            if not node.is_end_of_word:
                return False
            node.is_end_of_word = False
            return len(node.children) == 0
        char = word[depth]
        if char not in node.children:
            return False
        should_delete_child = _delete(node.children[char], word, depth + 1)
        if should_delete_child:
            del node.children[char]
            return len(node.children) == 0
        return False
    _delete(self.root, word, 0)
```