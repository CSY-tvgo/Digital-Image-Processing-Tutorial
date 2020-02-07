import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ex4_锐化 import *

# 实验四_额外：彩色图片锐化


# 使用拉普拉斯算子进行锐化的函数
# 传入参数： rawImage：原始的RGB图
#           alpha: 锐化程度
# 返回值为变换后的图
def sharpen_Laplacian_RGB(rawImage, alpha):
    aft = np.zeros(rawImage.shape, dtype='uint8')
    # 对三个通道分别进行拉普拉斯锐化
    aft[:, :, 0] = sharpen_Laplacian(rawImage[:, :, 0], alpha)
    aft[:, :, 1] = sharpen_Laplacian(rawImage[:, :, 1], alpha)
    aft[:, :, 2] = sharpen_Laplacian(rawImage[:, :, 2], alpha)
    return aft


# 直接执行该文件时运行下列代码
if __name__ == '__main__':
    # 导入图片
    raw = cv2.imread("images/ColorFlower.bmp")
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)  # 把BGR排布转换成RGB排布

    # 此部分为实际处理
    aft1 = np.zeros(raw.shape, dtype='uint8')
    aft1[:, :, 0] = sharpen_Laplacian(raw[:, :, 0], 1)

    aft2 = np.zeros(raw.shape, dtype='uint8')
    aft2[:, :, 1] = sharpen_Laplacian(raw[:, :, 1], 1)

    aft3 = np.zeros(raw.shape, dtype='uint8')
    aft3[:, :, 2] = sharpen_Laplacian(raw[:, :, 2], 1)

    aft4 = sharpen_Laplacian_RGB(raw, 1)

    # 以下代码用于显示图片
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

    plt.subplot(2, 2, 1)
    plt.imshow(aft1)
    plt.title("锐化之后的R通道")

    plt.subplot(2, 2, 2)
    plt.imshow(aft2)
    plt.title("锐化之后的G通道")

    plt.subplot(2, 2, 3)
    plt.imshow(aft3)
    plt.title("锐化之后B通道")

    plt.subplot(2, 2, 4)
    plt.imshow(aft4)
    plt.title("RGB锐化之后")

    plt.show()
