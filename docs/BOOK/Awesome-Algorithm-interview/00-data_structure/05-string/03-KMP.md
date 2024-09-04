> KMP


### KMP 算法

KMP（Knuth-Morris-Pratt）算法是一种高效的字符串匹配算法，可以在文本字符串中快速找到模式字符串的位置。KMP 算法通过预处理模式字符串，构建部分匹配表（Partial Match Table），避免重复的匹配过程，从而提高匹配效率。

### 算法原理

KMP 算法的核心思想是利用部分匹配表，在匹配失败时，利用已经匹配的信息尽量减少字符比较的次数。部分匹配表记录了模式字符串的前缀和后缀相同的最大长度。

#### 部分匹配表（Partial Match Table）

部分匹配表（也称为前缀函数，prefix function）是一个数组，用于记录模式字符串中每个位置的前缀和后缀相同的最大长度。

#### 构建部分匹配表

1. 初始化部分匹配表 `lps` 数组，长度与模式字符串相同，初始值均为0。
2. 使用两个指针 `len` 和 `i` 分别指向最长相同前缀后缀的结尾和当前字符。
3. 如果当前字符匹配，`len` 和 `i` 同时向前移动，并更新 `lps` 数组。
4. 如果不匹配，若 `len` 不为0，将 `len` 更新为 `lps[len-1]`；否则，`i` 向前移动，并将 `lps[i]` 置为0。

**代码实现**：

```python
def compute_lps(pattern):
    length = 0
    lps = [0] * len(pattern)
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

# 示例使用
pattern = "ABABCABAB"
print(compute_lps(pattern))  # 输出: [0, 0, 1, 2, 0, 1, 2, 3, 4]
```

#### KMP 匹配过程

1. 使用部分匹配表 `lps` 进行匹配：
   - 初始化 `i` 和 `j`，分别指向文本和模式的第一个字符。
   - 当 `i` 小于文本长度时：
     - 如果 `pattern[j]` 等于 `text[i]`，`i` 和 `j` 同时向前移动。
     - 如果 `j` 达到模式长度，说明匹配成功，记录匹配位置。
     - 如果不匹配，若 `j` 不为0，将 `j` 更新为 `lps[j-1]`；否则，`i` 向前移动。

**代码实现**：

```python
def kmp_search(text, pattern):
    m = len(pattern)
    n = len(text)
    lps = compute_lps(pattern)
    i = 0
    j = 0
    matches = []

    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return matches

# 示例使用
text = "ABABDABACDABABCABAB"
pattern = "ABABCABAB"
print(kmp_search(text, pattern))  # 输出: [10]
```

### 解释

1. **部分匹配表的构建**：
   - 使用 `compute_lps` 函数构建部分匹配表 `lps`，用于记录模式字符串中每个位置的前缀和后缀相同的最大长度。

2. **KMP 匹配过程**：
   - 使用 `kmp_search` 函数进行匹配，利用部分匹配表避免重复匹配，从而提高匹配效率。
   - 在匹配过程中，若字符匹配，指针 `i` 和 `j` 同时向前移动；若 `j` 达到模式长度，记录匹配位置；若字符不匹配，根据部分匹配表更新 `j`，继续匹配。

### 优点和缺点

**优点**：
- **高效**：时间复杂度为 O(m+n)，其中 m 为模式字符串长度，n 为文本字符串长度。
- **避免重复匹配**：利用部分匹配表，避免重复的字符比较，提高匹配效率。

**缺点**：
- **实现复杂**：相较于简单的匹配算法，实现较为复杂，需要构建部分匹配表。

### 参考文献

1. **Knuth, D. E., Morris, J. H., & Pratt, V. R. (1977). "Fast pattern matching in strings." SIAM Journal on Computing.**
   - 这篇论文介绍了 KMP 算法的基本原理和实现细节。

2. **GeeksforGeeks - KMP Algorithm for Pattern Searching**：
   - 提供了 KMP 算法的详细解释和代码实现。
   - [链接](https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/)

3. **LeetCode - Implement strStr()**：
   - 提供了使用 KMP 算法解决字符串匹配问题的详细描述和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/implement-strstr/)