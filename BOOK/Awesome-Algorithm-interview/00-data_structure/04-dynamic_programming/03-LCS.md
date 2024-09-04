> LCS


### 动态规划 - 最长公共子序列（LCS）

最长公共子序列（Longest Common Subsequence，LCS）问题是指在给定的两个序列中，找到它们最长的子序列，使得这个子序列同时是两个序列的子序列。LCS 是一种经典的动态规划问题，广泛应用于生物信息学、版本控制系统等领域。

### 算法步骤

1. **定义状态**：
   - 用 `dp[i][j]` 表示序列 `A` 的前 `i` 个元素和序列 `B` 的前 `j` 个元素的最长公共子序列长度。

2. **初始化**：
   - `dp[i][0] = 0`：表示任意序列和空序列的LCS长度为0。
   - `dp[0][j] = 0`：表示空序列和任意序列的LCS长度为0。

3. **状态转移**：
   - 如果 `A[i-1] == B[j-1]`，则 `dp[i][j] = dp[i-1][j-1] + 1`。
   - 如果 `A[i-1] != B[j-1]`，则 `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`。

4. **结果**：
   - `dp[m][n]` 即为 `A` 和 `B` 的最长公共子序列长度，其中 `m` 和 `n` 分别为 `A` 和 `B` 的长度。

### 代码实现

```python
def lcs(A, B):
    m, n = len(A), len(B)
    # 初始化dp数组
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 动态规划填表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 构建最长公共子序列
    i, j = m, n
    lcs_str = []
    while i > 0 and j > 0:
        if A[i - 1] == B[j - 1]:
            lcs_str.append(A[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return dp[m][n], ''.join(reversed(lcs_str))

# 示例使用
A = "ABCBDAB"
B = "BDCAB"
length, lcs_str = lcs(A, B)
print(f"LCS长度: {length}")
print(f"LCS: {lcs_str}")  # 输出: LCS长度: 4, LCS: BCAB
```

### 解释

1. **初始化**：
   - `dp[i][0] = 0` 和 `dp[0][j] = 0` 初始化了边界条件，表示任何序列与空序列的LCS长度为0。

2. **状态转移**：
   - 如果当前字符相等，则 `dp[i][j] = dp[i-1][j-1] + 1` 表示将当前字符加入LCS。
   - 如果当前字符不相等，则 `dp[i][j] = max(dp[i-1][j], dp[i][j-1])` 表示不将当前字符加入LCS，选择较长的子序列。

3. **构建LCS**：
   - 从 `dp[m][n]` 开始，根据 `dp` 数组的状态转移规则，逆向构建LCS。

### 时间复杂度和空间复杂度

- **时间复杂度**：O(m * n)，其中 m 和 n 分别为两个序列的长度。
- **空间复杂度**：O(m * n)，需要一个二维数组来存储中间状态。

### 参考文献

1. **GeeksforGeeks - Longest Common Subsequence**：
   - 提供了LCS问题的详细解释和实现示例。
   - [链接](https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/)

2. **LeetCode Problem - Longest Common Subsequence**：
   - 提供了LCS问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/longest-common-subsequence/)
