> 超大数相加


当处理超大数相加的问题时，普通的数据类型可能无法处理这些数值的存储和计算。为了解决这个问题，可以使用字符串或数组来模拟手动加法过程。以下是实现超大数相加的详细步骤和代码。

### 方法：逐位相加

1. **初始化**：
   - 使用两个指针分别指向两个字符串的末尾，表示最低位。
   - 初始化一个变量 `carry` 用于存储进位。

2. **逐位相加**：
   - 从最低位到最高位，逐位相加，并考虑进位。
   - 每一位的结果存储在结果列表中。

3. **处理剩余进位**：
   - 如果计算结束后还有进位，需要将进位添加到结果列表中。

4. **反转结果**：
   - 因为结果是从最低位开始存储的，需要反转结果列表得到最终结果。

### 代码实现

```python
def add_large_numbers(num1, num2):
    # 初始化指针和进位
    i, j = len(num1) - 1, len(num2) - 1
    carry = 0
    result = []

    # 逐位相加
    while i >= 0 or j >= 0 or carry:
        n1 = int(num1[i]) if i >= 0 else 0
        n2 = int(num2[j]) if j >= 0 else 0
        total = n1 + n2 + carry
        carry = total // 10
        result.append(total % 10)
        i -= 1
        j -= 1

    # 结果反转并转换为字符串
    return ''.join(map(str, result[::-1]))

# 示例使用
num1 = "123456789123456789"
num2 = "987654321987654321"
print(add_large_numbers(num1, num2))  # 输出: 1111111111111111110
```

### 解释

1. **初始化**：
   - `i` 和 `j` 分别指向 `num1` 和 `num2` 的最后一位，初始化 `carry` 为 0。

2. **逐位相加**：
   - 使用 `while` 循环遍历两个字符串，从最低位到最高位逐位相加。
   - 每次循环获取当前位的值，如果超出字符串长度则视为 0。
   - 计算当前位的总和 `total`，更新 `carry` 和当前位的结果。

3. **处理剩余进位**：
   - 循环结束后，如果 `carry` 不为 0，需要将 `carry` 添加到结果列表中。

4. **结果反转**：
   - 因为结果列表中的数字是从最低位到最高位存储的，最后需要反转结果列表并转换为字符串。

### 复杂度分析

- **时间复杂度**：O(n)，其中 n 是两个字符串的最大长度。每个位的相加和进位操作都是常数时间复杂度。
- **空间复杂度**：O(n)，用于存储结果的列表。

### 参考文献

1. **LeetCode Discuss - Add Strings**：
   - 提供了字符串相加问题的详细解释和解决方案。
   - [LeetCode链接](https://leetcode.com/problems/add-strings/)

2. **GeeksforGeeks - Sum of two large numbers**：
   - 提供了处理超大数相加的详细解释和代码示例。
   - [GeeksforGeeks链接](https://www.geeksforgeeks.org/sum-of-two-large-numbers/)