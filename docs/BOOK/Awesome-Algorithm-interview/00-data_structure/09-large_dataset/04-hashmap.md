> hashmap


哈希表（HashMap）是一种高效的数据结构，常用于快速插入、删除和查找操作。哈希表通过哈希函数将键映射到表中的位置，从而实现快速的数据访问。以下是哈希表的详细展开和实现步骤：

### 哈希表的特点
1. **哈希函数**：
   - 将输入的键映射为一个整数，这个整数通常是数组的索引。
   - 一个好的哈希函数应当尽量均匀分布键，以减少碰撞。

2. **碰撞处理**：
   - 当多个键映射到相同的位置时，需要解决冲突。
   - 常见的冲突解决方法包括链地址法（Separate Chaining）和开放地址法（Open Addressing）。

3. **操作**：
   - **插入**：将键值对插入到哈希表中。
   - **查找**：根据键查找对应的值。
   - **删除**：从哈希表中删除键值对。

### 哈希表的实现步骤

#### 步骤一：定义哈希表节点类
```python
class HashNode:
    def __init__(self, key, value):
        self.key = key  # 键
        self.value = value  # 值
        self.next = None  # 指向下一个节点的指针，用于解决冲突
```

#### 步骤二：定义哈希表类
```python
class HashMap:
    def __init__(self, capacity=100):
        self.capacity = capacity  # 哈希表的容量
        self.size = 0  # 当前存储的键值对数量
        self.table = [None] * self.capacity  # 哈希表，初始化为None

    def _hash(self, key):
        """哈希函数，将键映射为数组索引"""
        return hash(key) % self.capacity

    def insert(self, key, value):
        """插入键值对"""
        index = self._hash(key)
        new_node = HashNode(key, value)
        
        if self.table[index] is None:
            self.table[index] = new_node
        else:
            current = self.table[index]
            while current.next is not None:
                if current.key == key:
                    current.value = value  # 如果键已存在，则更新值
                    return
                current = current.next
            if current.key == key:
                current.value = value
            else:
                current.next = new_node
        self.size += 1

    def search(self, key):
        """查找值"""
        index = self._hash(key)
        current = self.table[index]
        
        while current is not None:
            if current.key == key:
                return current.value
            current = current.next
        return None

    def delete(self, key):
        """删除键值对"""
        index = self._hash(key)
        current = self.table[index]
        prev = None
        
        while current is not None:
            if current.key == key:
                if prev is None:
                    self.table[index] = current.next
                else:
                    prev.next = current.next
                self.size -= 1
                return True
            prev = current
            current = current.next
        return False
```

### 哈希表操作示例
```python
# 创建哈希表
hash_map = HashMap()

# 插入键值对
hash_map.insert("apple", 1)
hash_map.insert("banana", 2)
hash_map.insert("grape", 3)

# 查找值
print(hash_map.search("apple"))  # 输出: 1
print(hash_map.search("banana"))  # 输出: 2
print(hash_map.search("grape"))  # 输出: 3
print(hash_map.search("orange"))  # 输出: None

# 删除键值对
hash_map.delete("banana")
print(hash_map.search("banana"))  # 输出: None
```

### 代码详细解释
1. **HashNode类**：
   - `self.key`：存储键。
   - `self.value`：存储值。
   - `self.next`：指向下一个节点，用于处理冲突。

2. **HashMap类**：
   - `self.capacity`：哈希表的容量，默认为100。
   - `self.size`：当前存储的键值对数量。
   - `self.table`：哈希表的数组，初始化为None。
   - `_hash`方法：哈希函数，将键映射为数组索引。
   - `insert`方法：插入键值对，使用链地址法处理冲突。如果键已存在，则更新值。
   - `search`方法：查找值，遍历链表查找对应的键。
   - `delete`方法：删除键值对，遍历链表查找并删除对应的键。

### 进一步扩展
可以在哈希表的基础上实现更多功能，如：
- **动态扩容**：当负载因子（size/capacity）超过某个阈值时，动态扩容哈希表。
- **迭代器**：实现迭代器，用于遍历哈希表中的所有键值对。
- **自定义哈希函数**：允许用户传入自定义哈希函数，以适应不同的使用场景。

#### 动态扩容
```python
def _resize(self):
    """动态扩容哈希表"""
    old_table = self.table
    self.capacity *= 2
    self.table = [None] * self.capacity
    self.size = 0

    for node in old_table:
        while node is not None:
            self.insert(node.key, node.value)
            node = node.next

def insert(self, key, value):
    """插入键值对"""
    if self.size / self.capacity > 0.7:  # 负载因子大于0.7时扩容
        self._resize()

    index = self._hash(key)
    new_node = HashNode(key, value)
    
    if self.table[index] is None:
        self.table[index] = new_node
    else:
        current = self.table[index]
        while current.next is not None:
            if current.key == key:
                current.value = value
                return
            current = current.next
        if current.key == key:
            current.value = value
        else:
            current.next = new_node
    self.size += 1
```