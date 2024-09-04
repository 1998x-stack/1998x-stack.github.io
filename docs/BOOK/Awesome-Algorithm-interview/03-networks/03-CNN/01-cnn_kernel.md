> 给定卷积核的尺寸，特征图大小计算方法？


### 给定卷积核的尺寸，特征图大小计算方法

在卷积神经网络（CNN）中，计算特征图大小是一个重要的步骤，涉及到卷积核（filter）、步幅（stride）、填充（padding）等参数。以下是计算特征图大小的方法：

#### 公式

给定输入特征图的大小 $ W_{in} \times H_{in} $（宽度和高度），卷积核的大小 $ K \times K $（假设卷积核是方形的），步幅 $ S $ 和填充 $ P $，输出特征图的宽度 $ W_{out} $ 和高度 $ H_{out} $ 可以通过以下公式计算：

$$ W_{out} = \left\lfloor \frac{W_{in} - K + 2P}{S} \right\rfloor + 1 $$
$$ H_{out} = \left\lfloor \frac{H_{in} - K + 2P}{S} \right\rfloor + 1 $$

其中，$\left\lfloor \cdot \right\rfloor$ 表示向下取整。

#### 参数解释

1. **输入特征图大小 $ W_{in} \times H_{in} $**：
   - 输入特征图的宽度和高度。

2. **卷积核大小 $ K \times K $**：
   - 卷积核的宽度和高度。

3. **步幅 $ S $**：
   - 卷积核在输入特征图上滑动的步幅。步幅决定了卷积核每次移动的像素数。

4. **填充 $ P $**：
   - 输入特征图的填充大小。填充是为了在边缘添加额外的像素，使得卷积操作可以覆盖整个输入特征图。
   - 常见的填充方式有 `valid`（无填充）和 `same`（填充使输出特征图大小与输入特征图相同）。

#### 实例计算

假设有一个输入特征图大小为 $ 32 \times 32 $，卷积核大小为 $ 3 \times 3 $，步幅为 1，填充为 1。计算输出特征图的大小：

1. **计算输出特征图的宽度 $ W_{out} $**：
   $$
   W_{out} = \left\lfloor \frac{32 - 3 + 2 \cdot 1}{1} \right\rfloor + 1 = \left\lfloor \frac{32}{1} \right\rfloor + 1 = 32
   $$

2. **计算输出特征图的高度 $ H_{out} $**：
   $$
   H_{out} = \left\lfloor \frac{32 - 3 + 2 \cdot 1}{1} \right\rfloor + 1 = \left\lfloor \frac{32}{1} \right\rfloor + 1 = 32
   $$

所以，输出特征图的大小为 $ 32 \times 32 $。

#### 不同填充方式的计算

- **无填充（valid）**：
  $$
  P = 0
  $$
  例如，输入特征图大小为 $ 32 \times 32 $，卷积核大小为 $ 3 \times 3 $，步幅为 1，填充为 0。
  $$
  W_{out} = \left\lfloor \frac{32 - 3 + 0}{1} \right\rfloor + 1 = \left\lfloor \frac{29}{1} \right\rfloor + 1 = 30
  $$
  $$
  H_{out} = \left\lfloor \frac{32 - 3 + 0}{1} \right\rfloor + 1 = \left\lfloor \frac{29}{1} \right\rfloor + 1 = 30
  $$
  输出特征图的大小为 $ 30 \times 30 $。

- **相同填充（same）**：
  填充的目的是保持输出特征图的大小与输入特征图相同。在这种情况下，填充 $ P $ 的计算为：
  $$
  P = \frac{K - 1}{2}
  $$
  例如，输入特征图大小为 $ 32 \times 32 $，卷积核大小为 $ 3 \times 3 $，步幅为 1。
  $$
  P = \frac{3 - 1}{2} = 1
  $$
  输出特征图的大小为 $ 32 \times 32 $。

### 参考资料

1. [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)
2. [Deep Learning Book](http://www.deeplearningbook.org/)
3. [Towards Data Science: Understanding the Calculation of Output Size in Convolutional Neural Networks](https://towardsdatascience.com/understanding-the-calculation-of-output-size-in-convolutional-neural-networks-cb03103b9a42)