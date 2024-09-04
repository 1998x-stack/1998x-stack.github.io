> 八皇后，全排列，组合

### 搜索回溯：八皇后问题，全排列，组合

搜索回溯是一种常用的算法技术，用于解决一些具有复杂约束的组合问题。通过回溯算法，可以在构造解的过程中尽早剪枝，减少计算量。以下详细介绍八皇后问题，全排列和组合的回溯算法。

### 1. 八皇后问题

八皇后问题是经典的N皇后问题的特例，在8x8的棋盘上放置8个皇后，使得它们彼此不受攻击。皇后可以攻击同一行、同一列或对角线上的任何棋子。

**算法步骤**：
1. 从第一行开始，在每一行尝试放置一个皇后。
2. 对于当前行的每一列，判断是否可以安全地放置皇后（即该位置不受之前放置的皇后攻击）。
3. 如果可以放置，递归地尝试放置下一行的皇后。
4. 如果成功放置了所有的皇后，记录当前解。
5. 回溯，尝试下一列。

**实现代码**：
```python
def solve_n_queens(n):
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                return False
        return True
    
    def solve(board, row):
        if row == n:
            result.append(board[:])
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                solve(board, row + 1)
                board[row] = -1
    
    result = []
    board = [-1] * n
    solve(board, 0)
    return result

# 示例使用
n = 8
solutions = solve_n_queens(n)
print(len(solutions))  # 输出: 92
```

### 2. 全排列

全排列问题是指给定一个不重复的序列，返回其所有可能的全排列。

**算法步骤**：
1. 从序列的第一个元素开始，依次将每一个元素固定在当前位置。
2. 对剩余的元素递归地生成全排列。
3. 回溯，将固定的元素交换回来，尝试下一种排列。

**实现代码**：
```python
def permute(nums):
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]
    
    result = []
    backtrack(0)
    return result

# 示例使用
nums = [1, 2, 3]
permutations = permute(nums)
print(permutations)  # 输出: [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 2, 1], [3, 1, 2]]
```

### 3. 组合

组合问题是指从给定的一组数中，选出指定数量的元素，返回所有可能的组合。

**算法步骤**：
1. 从第一个元素开始，依次选择每一个元素作为组合的一部分。
2. 对剩余的元素递归地生成组合。
3. 回溯，移除当前选择的元素，尝试下一种组合。

**实现代码**：
```python
def combine(n, k):
    def backtrack(start, comb):
        if len(comb) == k:
            result.append(comb[:])
            return
        for i in range(start, n + 1):
            comb.append(i)
            backtrack(i + 1, comb)
            comb.pop()
    
    result = []
    backtrack(1, [])
    return result

# 示例使用
n = 4
k = 2
combinations = combine(n, k)
print(combinations)  # 输出: [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
```

### 总结

- **八皇后问题**：通过回溯和剪枝策略，有效地放置N个皇后，使得它们互不攻击。
- **全排列问题**：使用递归和交换技术生成所有可能的排列。
- **组合问题**：通过递归和回溯，生成所有可能的组合。
