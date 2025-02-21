# nn
从零开始编写一个全连接神经网络，并进行优化。试试看可以将性能优化到什么程度。

## 数据集
本项目使用[MINIST数据集](https://yann.lecun.com/exdb/mnist/)。MNIST 数据集由 Yann LeCun、Léon Bottou、Yoshua Bengio 和 Patrick Haffner 提供。它最初发表于以下论文：
LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
这是一个广泛用于机器学习和计算机视觉领域的手写数字图像数据集。MNIST 数据集包含 60,000 张训练图像和 10,000 张测试图像，每张图像为 28x28 像素的灰度图像，标注了对应的数字类别（0-9）。

## 数据集预处理
为了避免在代码中使用多余的库，所以提前将图片文件处理为文本文件（TXT）。文件中第一行是该图片显示的数字。文件名命名格式为{test/train}+{序号}。文本内容为图片像素点的灰度值，每个数值占一行，顺序为图片文件按行读取。