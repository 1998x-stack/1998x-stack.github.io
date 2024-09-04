### 常见排序相关问题

排序算法是计算机科学中重要的基础知识之一。除了常见的排序算法之外，还有许多排序相关的问题和概念，下面介绍一些常见的问题及其解答。

#### 1. 稳定排序和不稳定排序

**问题**：什么是稳定排序和不稳定排序？例子有哪些？

**解答**：
- **稳定排序**：如果两个相等的元素在排序前后的相对位置保持不变，那么该排序算法是稳定的。常见的稳定排序算法有：
  - 插入排序
  - 归并排序
  - 冒泡排序
  - 计数排序
  - 基数排序

- **不稳定排序**：如果两个相等的元素在排序前后的相对位置可能会发生改变，那么该排序算法是不稳定的。常见的不稳定排序算法有：
  - 快速排序
  - 选择排序
  - 堆排序

**参考文献**：
- [GeeksforGeeks - Stable and Unstable Sorting Algorithms](https://www.geeksforgeeks.org/stable-and-unstable-sorting-algorithms/)

#### 2. 内部排序和外部排序

**问题**：什么是内部排序和外部排序？

**解答**：
- **内部排序**：当所有需要排序的记录都被加载到内存中进行排序时，称为内部排序。常见的内部排序算法包括快速排序、归并排序和堆排序等。
- **外部排序**：当需要排序的记录数量非常大，无法全部加载到内存中时，必须使用外存（如磁盘）来辅助排序，称为外部排序。常见的外部排序算法有多路归并排序和磁盘排序等。

**参考文献**：
- [Wikipedia - External Sorting](https://en.wikipedia.org/wiki/External_sorting)

#### 3. 最好情况下的排序算法

**问题**：在最好情况下，哪些排序算法的时间复杂度是 $O(n)$？

**解答**：
- **计数排序**：如果输入的元素都是整数且范围已知，则计数排序可以在 $O(n + k)$ 时间内完成，其中 $k$ 是输入数据范围。
- **基数排序**：对于位数固定的整数，可以在 $O(nk)$ 时间内完成排序，其中 $k$ 是整数的位数。
- **桶排序**：如果输入数据均匀分布，可以在 $O(n + k)$ 时间内完成排序，其中 $k$ 是桶的数量。

**参考文献**：
- [GeeksforGeeks - Counting Sort](https://www.geeksforgeeks.org/counting-sort/)
- [GeeksforGeeks - Radix Sort](https://www.geeksforgeeks.org/radix-sort/)
- [GeeksforGeeks - Bucket Sort](https://www.geeksforgeeks.org/bucket-sort-2/)

#### 4. 适用于链表的排序算法

**问题**：哪些排序算法适用于链表？

**解答**：
- **归并排序**：归并排序非常适合链表，因为它不需要随机访问数据，可以通过递归实现链表的排序。
- **插入排序**：对于几乎有序的链表，插入排序非常高效，并且易于实现。

**参考文献**：
- [GeeksforGeeks - Merge Sort for Linked List](https://www.geeksforgeeks.org/merge-sort-for-linked-list/)
- [GeeksforGeeks - Insertion Sort for Singly Linked List](https://www.geeksforgeeks.org/insertion-sort-for-singly-linked-list/)

#### 5. 时间复杂度为 $O(n \log n)$ 的排序算法

**问题**：哪些排序算法的时间复杂度是 $O(n \log n)$？

**解答**：
- **快速排序**：平均时间复杂度为 $O(n \log n)$。
- **归并排序**：时间复杂度为 $O(n \log n)$。
- **堆排序**：时间复杂度为 $O(n \log n)$。

**参考文献**：
- [Wikipedia - Quicksort](https://en.wikipedia.org/wiki/Quicksort)
- [Wikipedia - Merge Sort](https://en.wikipedia.org/wiki/Merge_sort)
- [Wikipedia - Heapsort](https://en.wikipedia.org/wiki/Heapsort)