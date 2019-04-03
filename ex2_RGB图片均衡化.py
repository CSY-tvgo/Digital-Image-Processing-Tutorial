import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ex2_均衡化 import *


# 实验二_额外的额外：彩色图像的直方图均衡化


# 用于直方图均衡化的函数，传入参数为原始的RGB图，返回值为变换后的图
def HistEqualize_RGB(rawImage):
    aft = np.zeros(rawImage.shape, dtype='uint8')
    aft[:, :, 0] = HistEqualize(rawImage[:, :, 0])
    aft[:, :, 1] = HistEqualize(rawImage[:, :, 1])
    aft[:, :, 2] = HistEqualize(rawImage[:, :, 2])  # 对三个通道分别进行直方图均衡化
    return aft


# 直接执行该文件时运行下列代码
if __name__ == '__main__':
    # 导入图片
    raw = cv2.imread("./RGBPIC1.bmp")

    # 此部分为实际处理
    aft1 = HistEqualize_RGB(raw)

    # 以下代码用于显示图片
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

    plt.figure(figsize=(12, 9), dpi=100)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    plt.title("原始图像")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(aft1, cv2.COLOR_BGR2RGB))
    plt.title("直方图均衡化之后")

    plt.show()
