# nn
从零开始编写一个全连接神经网络，并进行优化。试试看可以将性能优化到什么程度。

## 数据集
本项目使用[MINIST数据集](https://yann.lecun.com/exdb/mnist/)。MNIST 数据集由 Yann LeCun、Léon Bottou、Yoshua Bengio 和 Patrick Haffner 提供。它最初发表于以下论文：
LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

这是一个广泛用于机器学习和计算机视觉领域的手写数字图像数据集。MNIST 数据集包含 60,000 张训练图像和 10,000 张测试图像，每张图像为 28x28 像素的灰度图像，标注了对应的数字类别（0-9）。

## 数据集预处理
为了避免在代码中使用多余的库来对图片进行处理，所以提前将图片文件处理为文本文件（TXT）。

数据集总共包含两个文件：train_1.txt、train_1.txt和 test.txt。其中train_1.txt、train_1.txt为训练集，由于 GitHub 文件大小限制在100M，所以切割为两个文件，test.txt 为测试集。

文件内每行为一组数据，共785个数据值，使用空格隔开。每组第一个数值为改组图片显示的数字，结果为0-9。剩下784个数值为图片转换后的数值。数值内容为图片像素点的灰度值，顺序为图片文件按行顺序读取。