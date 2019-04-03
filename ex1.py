import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# 实验一：画直方图


# 为便于后续实验的使用，这里声明一个函数，用于计算直方图，传入参数为原灰度图，返回参数为计算后的直方图
def caculateHist(rawImage):
    height = rawImage.shape[0]  # 获取图像的高度
    width = rawImage.shape[1]  # 获取图像的宽度
    hist = np.zeros(256)  # 初始化一个长度为256、值全为0的向量
    for r in range(0, height):  # 遍历每个像素
        for c in range(0, width):
            grey = rawImage[r][c]  # 获取灰度
            hist[grey] = hist[grey]+1  # 将对应灰度的记录数量+1
    hist = hist/rawImage.size  # 将hist向量点除原图的像素数量（即归一化）
    return hist


# 直接执行该文件时运行下列代码
if __name__ == '__main__':
    # 导入图片
    raw = cv2.imread("./balloon.bmp", 0)  # 0表示以灰度图形式导入，导入后raw是一个二维矩阵

    # 实际处理
    hist = caculateHist(raw)  # 调用前面声明的函数来计算直方图

    # 以下代码用于显示图片
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

    plt.subplot(1, 2, 1)  # 显示原图
    plt.imshow(cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB))
    plt.title("原始图像")

    plt.subplot(1, 2, 2)  # 显示直方图
    plt.bar(range(0, 256), hist)
    plt.title("直方图")

    plt.show()
