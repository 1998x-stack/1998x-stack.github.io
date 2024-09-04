> 给定字符串是否是数字？


判断一个字符串是否是数字是一个常见的问题，特别是在数据输入验证和数据清理过程中。我们可以使用多种方法来检查字符串是否是数字，包括正则表达式和简单的迭代检查。下面是一些方法来实现这一功能。

### 方法一：使用内置方法

Python 提供了一些内置方法，可以帮助我们简单快速地判断一个字符串是否是数字。

1. **str.isdigit()**：
   - 该方法返回 True 如果字符串只包含数字字符（0-9），否则返回 False。这个方法不处理负号、小数点或科学计数法。
   
```python
def is_number(s):
    return s.isdigit()

# 示例使用
print(is_number("123"))  # 输出: True
print(is_number("-123"))  # 输出: False
print(is_number("123.45"))  # 输出: False
```

2. **尝试转换为浮点数**：
   - 使用 try-except 结构尝试将字符串转换为浮点数。如果转换成功，字符串是数字；否则，抛出异常。
   
```python
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# 示例使用
print(is_number("123"))  # 输出: True
print(is_number("-123"))  # 输出: True
print(is_number("123.45"))  # 输出: True
print(is_number("123.45.67"))  # 输出: False
```

### 方法二：使用正则表达式

正则表达式提供了一种灵活而强大的方式来匹配复杂的字符串模式。我们可以使用正则表达式来匹配整数、小数和科学计数法表示的数字。

```python
import re

def is_number(s):
    pattern = re.compile(r'^-?\d+(\.\d+)?(e[+-]?\d+)?$')
    return bool(pattern.match(s))

# 示例使用
print(is_number("123"))  # 输出: True
print(is_number("-123"))  # 输出: True
print(is_number("123.45"))  # 输出: True
print(is_number("1e10"))  # 输出: True
print(is_number("123.45.67"))  # 输出: False
```

### 解释

1. **内置方法**：
   - `str.isdigit()` 只能检查整数且没有负号或小数点。
   - 使用 `float()` 函数的 try-except 结构可以处理整数、小数和负数，但不能直接处理科学计数法。

2. **正则表达式**：
   - `^-?\d+(\.\d+)?(e[+-]?\d+)?$` 解释：
     - `^-?` 匹配可选的负号。
     - `\d+` 匹配一个或多个数字。
     - `(\.\d+)?` 可选的小数部分。
     - `(e[+-]?\d+)?$` 可选的科学计数法部分。

### 优缺点

**优点**：
- **内置方法**：简单易用，性能高。
- **正则表达式**：灵活，能够处理更复杂的数字格式。

**缺点**：
- **内置方法**：对格式限制严格，不能处理科学计数法。
- **正则表达式**：复杂度高，学习成本较高。

### 参考文献

1. **Python 官方文档**：
   - 提供了 `str.isdigit()` 和其他字符串方法的详细说明。
   - [链接](https://docs.python.org/3/library/stdtypes.html#str.isdigit)
   
2. **GeeksforGeeks - Python program to check if a string is numeric**：
   - 提供了多种方法来检查字符串是否是数字。
   - [链接](https://www.geeksforgeeks.org/python-program-to-check-if-a-string-is-numeric/)
