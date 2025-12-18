# nn
## 介绍
本项目计划从零开始编写一个全连接神经网络，并通过一系列策略对其进行优化。试试看可以将性能优化到什么程度。

### 编译环境
- C++11 及以上标准
- CMake 3.10 及以上

### 网络结构
当前实现的神经网络结构如下：
- 输入层：784 个节点（对应 28x28 像素的图像）
- 隐藏层：300 个节点
- 输出层：10 个节点（对应 0-9 数字类别）

## 数据集说明
本项目使用[MNIST数据集](https://yann.lecun.com/exdb/mnist/)。MNIST 数据集由 Yann LeCun、Léon Bottou、Yoshua Bengio 和 Patrick Haffner 提供。它最初发表于以下论文：
LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

### 数据集规模
这是一个广泛用于机器学习和计算机视觉领域的手写数字图像数据集，具体规格如下：
- 训练集：60,000 张手写数字图像
- 测试集：10,000 张手写数字图像
- 图像规格：28×28 像素
- 图像内容：0~9 数字类别

### 数据集预处理
为了避免在代码中使用多余的库来对图片进行处理，所以提前将图片文件处理为文本文件（TXT），具体说明如下：
- 数据文件：包含train.txt（训练集）和test.txt（测试集）两个核心文件
- 数据格式：每行对应一组完整样本，共 785 个数值，以空格分隔
- 数值含义：首项为样本标注（数字 0~9），后续 784 项为图像按行扫描得到的像素灰度值，直接用于模型输入

### 运行未优化项目（v1.0）
```
PS D:\Project\Github\du-xiang\fnn\build> mingw32-make
[ 20%] Building CXX object CMakeFiles/fnn.dir/FullConnNN.cpp.obj
[ 40%] Linking CXX executable fnn.exe
[100%] Built target fnn
PS D:\Project\Github\du-xiang\fnn\build> ./fnn.exe
** begins training **
progressing[==================================================] 60000/60000 it

time: 576278ms
```