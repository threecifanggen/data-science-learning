# 5. 重抽样方法

## 5.1 交叉验证(Cross-Validation)

### 5.1.1 验证集方法

使用一个比例的验证集(*Validation Set*)

### 5.1.2 Leave-One-Out Cross-Validation

每次取出一个做验证，然后用平均值作为结果：

$$
CV_{(n)} = \frac 1 n \sum_{i=1}^n {MSE}_i
$$

### 5.1.3 k-Fold Cross-Valisdation


$$
CV_{(k)} = \frac 1 k \sum^k_{i=1} {MSE}_i
$$

![](images/2021-10-29-13-40-46.png)

![](images/2021-10-29-13-56-38.png)

## 5.2 The Bootstrap


