> 图的搜索


### 图的搜索

图的搜索是一类在图数据结构上进行遍历和搜索的算法。图搜索算法主要有两类：深度优先搜索（Depth-First Search, DFS）和广度优先搜索（Breadth-First Search, BFS）。这些算法广泛应用于路径查找、连通性检测、图遍历等问题中。

### 深度优先搜索（DFS）

深度优先搜索是一种从图的起始节点出发，沿着每个分支尽可能深入搜索的算法。它可以用递归或显式栈来实现。

#### 递归实现

```python
def dfs_recursive(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start)  # 处理节点
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)
    return visited

# 示例使用
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
visited_nodes = dfs_recursive(graph, 'A')
```

#### 栈实现

```python
def dfs_stack(graph, start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            print(node)  # 处理节点
            stack.extend(neighbor for neighbor in graph[node] if neighbor not in visited)
    return visited

# 示例使用
visited_nodes = dfs_stack(graph, 'A')
```

### 广度优先搜索（BFS）

广度优先搜索是一种从图的起始节点出发，逐层广度优先地搜索图的算法。它通常用队列来实现。

#### 队列实现

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            print(node)  # 处理节点
            queue.extend(neighbor for neighbor in graph[node] if neighbor not in visited)
    return visited

# 示例使用
visited_nodes = bfs(graph, 'A')
```

### 图的搜索应用

1. **路径查找**：
   - DFS 和 BFS 都可以用于查找从起点到终点的路径。
   - BFS 更适合用于查找最短路径。

2. **连通性检测**：
   - 可以用 DFS 或 BFS 来检测图是否连通，或者找到图中的所有连通分量。

3. **拓扑排序**：
   - 可以用 DFS 来实现拓扑排序，用于有向无环图（DAG）。

### 比较 DFS 和 BFS

| 特点                  | 深度优先搜索（DFS）                              | 广度优先搜索（BFS）                          |
|-----------------------|--------------------------------------------------|----------------------------------------------|
| 数据结构              | 栈（递归或显式栈）                               | 队列                                          |
| 搜索策略              | 尽可能深度搜索                                   | 逐层广度搜索                                  |
| 适用场景              | 深度优先遍历、拓扑排序、找到所有解               | 最短路径查找、广度优先遍历                    |
| 记忆体使用            | 较小（仅需记录当前路径）                         | 较大（需要记录所有已访问节点）                |
| 完备性                | 可能陷入死循环，需要检测访问过的节点             | 保证找到最短路径，但需要检测访问过的节点      |

### 参考文献

1. **GeeksforGeeks - Graph and its representations**：
   - 详细介绍了图的表示方法和搜索算法。
   - [链接](https://www.geeksforgeeks.org/graph-and-its-representations/)

2. **LeetCode Discuss - Depth-First Search**：
   - 提供了关于深度优先搜索的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/tag/depth-first-search/)

3. **GeeksforGeeks - Breadth-First Search**：
   - 提供了关于广度优先搜索的详细描述和实现方法。
   - [链接](https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/)
