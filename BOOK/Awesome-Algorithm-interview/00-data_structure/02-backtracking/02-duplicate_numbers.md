> 重复数字的排列，重复数字的组合


### 含重复数字的排列

含重复数字的排列问题是指给定一个包含重复数字的序列，生成所有可能的排列，并确保排列的结果中没有重复的排列。可以通过回溯算法来实现，并在过程中使用一个集合来避免重复。

#### 代码实现

```python
def permute_unique(nums):
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] not in seen:
                seen.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start + 1)
                nums[start], nums[i] = nums[i], nums[start]
    
    result = []
    nums.sort()
    backtrack(0)
    return result

# 示例使用
nums = [1, 1, 2]
permutations = permute_unique(nums)
print(permutations)  # 输出: [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
```

### 含重复数字的组合

含重复数字的组合问题是指给定一个包含重复数字的序列，生成所有可能的组合，并确保组合的结果中没有重复的组合。可以通过回溯算法来实现，并在过程中避免选择相同的数字。

#### 代码实现

```python
def combine_unique(nums, k):
    def backtrack(start, comb):
        if len(comb) == k:
            result.append(comb[:])
            return
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            comb.append(nums[i])
            backtrack(i + 1, comb)
            comb.pop()
    
    result = []
    nums.sort()
    backtrack(0, [])
    return result

# 示例使用
nums = [1, 2, 2]
k = 2
combinations = combine_unique(nums, k)
print(combinations)  # 输出: [[1, 2], [2, 2]]
```

### 总结

- **含重复数字的排列**：通过回溯和集合避免重复生成所有可能的排列。
- **含重复数字的组合**：通过回溯和剪枝避免重复生成所有可能的组合。