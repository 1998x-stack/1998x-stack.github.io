### 经典的动态规划问题

动态规划（Dynamic Programming，DP）是一种非常强大的算法设计范式，用于解决具有重叠子问题和最优子结构性质的问题。除了背包问题和最长公共子序列（LCS），以下是一些其他经典的动态规划问题：

#### 1. 斐波那契数列（Fibonacci Sequence）

**问题描述**：计算斐波那契数列的第 n 项。斐波那契数列的定义如下：
$$ F(n) = F(n-1) + F(n-2) $$
$$ F(0) = 0, F(1) = 1 $$

**代码实现**：
```python
def fibonacci(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

# 示例使用
print(fibonacci(10))  # 输出: 55
```

#### 2. 最长递增子序列（Longest Increasing Subsequence, LIS）

**问题描述**：找到一个给定序列中最长递增子序列的长度。

**代码实现**：
```python
def length_of_lis(nums):
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# 示例使用
nums = [10, 9, 2, 5, 3, 7, 101, 18]
print(length_of_lis(nums))  # 输出: 4
```

#### 3. 最短路径问题（Shortest Path Problem）

**问题描述**：在一个有向图中找到从起点到终点的最短路径。

**代码实现**（Dijkstra算法的动态规划实现）：
```python
import heapq

def dijkstra(graph, start):
    min_heap = [(0, start)]
    dist = {start: 0}
    while min_heap:
        current_dist, node = heapq.heappop(min_heap)
        if current_dist > dist[node]:
            continue
        for neighbor, weight in graph[node]:
            distance = current_dist + weight
            if neighbor not in dist or distance < dist[neighbor]:
                dist[neighbor] = distance
                heapq.heappush(min_heap, (distance, neighbor))
    return dist

# 示例使用
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}
print(dijkstra(graph, 'A'))  # 输出: {'A': 0, 'B': 1, 'C': 3, 'D': 4}
```

#### 4. 三角形最小路径和（Triangle Minimum Path Sum）

**问题描述**：给定一个三角形，找出从顶部到底部的最小路径和。每一步可以移动到下一行相邻的数字。

**代码实现**：
```python
def minimum_total(triangle):
    if not triangle:
        return 0
    dp = triangle[-1]
    for i in range(len(triangle) - 2, -1, -1):
        for j in range(len(triangle[i])):
            dp[j] = triangle[i][j] + min(dp[j], dp[j + 1])
    return dp[0]

# 示例使用
triangle = [
    [2],
    [3, 4],
    [6, 5, 7],
    [4, 1, 8, 3]
]
print(minimum_total(triangle))  # 输出: 11
```

#### 5. 零钱兑换（Coin Change）

**问题描述**：给定不同面额的硬币和一个总金额，计算可以凑成总金额的最少硬币个数。如果不可能凑成总金额，返回 -1。

**代码实现**：
```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

# 示例使用
coins = [1, 2, 5]
amount = 11
print(coin_change(coins, amount))  # 输出: 3
```

### 参考文献

1. **LeetCode**：
   - 提供了大量经典的动态规划问题以及其解决方案。
   - [LeetCode](https://leetcode.com/)
2. **GeeksforGeeks**：
   - 详细介绍了多种经典动态规划问题及其实现。
   - [GeeksforGeeks](https://www.geeksforgeeks.org/fundamentals-of-algorithms/)
3. **Introduction to Algorithms by Cormen, Leiserson, Rivest, and Stein**：
   - 提供了全面的动态规划问题的理论和例子。
   - [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms)
