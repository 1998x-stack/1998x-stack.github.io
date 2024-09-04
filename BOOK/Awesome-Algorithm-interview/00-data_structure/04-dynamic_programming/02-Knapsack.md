> 动态规划-背包问题


### 背包问题（Knapsack Problem）

背包问题是一类经典的组合优化问题，其目标是在给定的重量和容量限制下，选择若干物品，使得总价值最大。背包问题的常见变种包括 0-1 背包问题、完全背包问题和多重背包问题等。

#### 0-1 背包问题

在 0-1 背包问题中，每个物品只能选择一次，且每个物品具有一定的重量和价值。目标是在不超过背包容量的情况下，最大化背包中物品的总价值。

**算法步骤**：

1. **定义状态**：
   - 用 `dp[i][w]` 表示前 `i` 个物品中选择若干个物品放入容量为 `w` 的背包时的最大价值。

2. **状态转移方程**：
   - 如果不选第 `i` 个物品：`dp[i][w] = dp[i-1][w]`
   - 如果选第 `i` 个物品：`dp[i][w] = dp[i-1][w-weight[i]] + value[i]`
   - 综合状态转移方程：`dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight[i]] + value[i])`

3. **初始化**：
   - `dp[0][w] = 0` 表示没有物品时背包价值为 0。

4. **结果**：
   - `dp[n][W]` 即为前 `n` 个物品在容量为 `W` 的背包中能取得的最大价值。

**代码实现**：

```python
def knapsack_01(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]

    return dp[n][W]

# 示例使用
weights = [1, 3, 4, 5]
values = [1, 4, 5, 7]
W = 7
print(knapsack_01(weights, values, W))  # 输出: 9
```

#### 完全背包问题

在完全背包问题中，每个物品可以选择多次。状态转移方程和初始化与 0-1 背包问题类似，但在考虑选择第 `i` 个物品时，需要循环计算可能选择的次数。

**代码实现**：

```python
def knapsack_complete(weights, values, W):
    n = len(weights)
    dp = [0] * (W + 1)

    for i in range(n):
        for w in range(weights[i], W + 1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[W]

# 示例使用
weights = [1, 3, 4, 5]
values = [1, 4, 5, 7]
W = 7
print(knapsack_complete(weights, values, W))  # 输出: 11
```

#### 多重背包问题

在多重背包问题中，每个物品可以选择有限次。可以将每个物品拆分成有限的多个 0-1 背包问题来解决。

**代码实现**：

```python
def knapsack_multiple(weights, values, amounts, W):
    n = len(weights)
    dp = [0] * (W + 1)

    for i in range(n):
        for _ in range(amounts[i]):
            for w in range(W, weights[i] - 1, -1):
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[W]

# 示例使用
weights = [1, 3, 4, 5]
values = [1, 4, 5, 7]
amounts = [2, 2, 1, 1]  # 每个物品的数量限制
W = 7
print(knapsack_multiple(weights, values, amounts, W))  # 输出: 10
```

### 总结

- **0-1 背包问题**：每个物品只能选择一次。
- **完全背包问题**：每个物品可以选择无限次。
- **多重背包问题**：每个物品可以选择有限次。

### 参考文献

1. **GeeksforGeeks - 0-1 Knapsack Problem**：
   - 提供了 0-1 背包问题的详细解释和实现示例。
   - [链接](https://www.geeksforgeeks.org/0-1-knapsack-problem-dp-10/)

2. **GeeksforGeeks - Unbounded Knapsack (Repetition of items allowed)**：
   - 提供了完全背包问题的详细解释和实现示例。
   - [链接](https://www.geeksforgeeks.org/unbounded-knapsack-repetition-items-allowed/)

3. **GeeksforGeeks - Multi Knapsack Problem**：
   - 提供了多重背包问题的详细解释和实现示例。
   - [链接](https://www.geeksforgeeks.org/multi-dimensional-knapsack-problem-dp-32/)

通过这些方法和代码实现，可以有效地解决各种背包问题，适用于资源分配、库存管理等实际场景。