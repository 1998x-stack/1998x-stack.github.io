> 动态规划-编辑距离



### 动态规划 - 编辑距离

编辑距离（Edit Distance），也称为Levenshtein距离，是指两个字符串之间通过插入、删除或替换字符而需要的最少编辑操作次数。编辑距离在自然语言处理、拼写检查、DNA序列比对等领域有广泛应用。

### 算法步骤

编辑距离问题可以通过动态规划来解决，具体步骤如下：

1. **定义状态**：
   - 用 `dp[i][j]` 表示将字符串 `word1[0...i-1]` 转换为 `word2[0...j-1]` 所需的最少编辑操作次数。

2. **初始化**：
   - `dp[0][j]` 表示将空字符串转换为 `word2[0...j-1]`，需要插入 `j` 次。
   - `dp[i][0]` 表示将 `word1[0...i-1]` 转换为空字符串，需要删除 `i` 次。

3. **状态转移**：
   - 如果 `word1[i-1] == word2[j-1]`，则 `dp[i][j] = dp[i-1][j-1]`。
   - 如果 `word1[i-1] != word2[j-1]`，则需要考虑三种操作：插入、删除、替换，选择其中最小的：
     $$
     dp[i][j] = \min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1)
     $$

4. **结果**：
   - `dp[m][n]` 即为将 `word1` 转换为 `word2` 所需的最少编辑操作次数，其中 `m` 和 `n` 分别为 `word1` 和 `word2` 的长度。

### 代码实现

```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    # 初始化dp数组
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化边界条件
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 动态规划填表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,    # 删除
                               dp[i][j - 1] + 1,    # 插入
                               dp[i - 1][j - 1] + 1) # 替换

    return dp[m][n]

# 示例使用
word1 = "horse"
word2 = "ros"
print(min_distance(word1, word2))  # 输出: 3
```

### 解释

1. **初始化**：
   - `dp[0][j] = j`：将空字符串转换为长度为 `j` 的字符串，需要 `j` 次插入。
   - `dp[i][0] = i`：将长度为 `i` 的字符串转换为空字符串，需要 `i` 次删除。

2. **状态转移**：
   - 如果 `word1[i-1] == word2[j-1]`，则当前位置字符相同，不需要额外操作，`dp[i][j] = dp[i-1][j-1]`。
   - 如果 `word1[i-1] != word2[j-1]`，则需要考虑插入、删除、替换三种操作，取最小值。

3. **返回结果**：
   - `dp[m][n]` 即为最终结果，表示将 `word1` 转换为 `word2` 所需的最少编辑操作次数。

### 时间复杂度和空间复杂度

- **时间复杂度**：O(m * n)，其中 m 和 n 分别为两个字符串的长度。
- **空间复杂度**：O(m * n)，需要一个二维数组来存储中间状态。

### 参考文献

1. **Levenshtein, Vladimir I. (1966). "Binary codes capable of correcting deletions, insertions, and reversals." Soviet Physics Doklady.**
   - 介绍了编辑距离的基础理论。

2. **GeeksforGeeks - Edit Distance**：
   - 提供了关于编辑距离问题的详细解释和实现示例。
   - [链接](https://www.geeksforgeeks.org/edit-distance-dp-5/)

3. **LeetCode Problem - Edit Distance**：
   - 提供了编辑距离问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/edit-distance/)