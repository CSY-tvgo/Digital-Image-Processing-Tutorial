import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# 实验六：伪彩色增强

# 把值抑制在0~255内
def suppress(num):
    if num > 255:
        return 255
    elif num < 0:
        return 0
    else:
        return num


# 用于伪彩色增强的函数
def fakeRGB(rawImage):
    raw_height = rawImage.shape[0]  # 获取原图像的高度
    raw_width = rawImage.shape[1]  # 获取原图像的宽度

    aftR = np.zeros((raw_height, raw_width, 3),
                    dtype='uint8')
    aftG = np.zeros((raw_height, raw_width, 3),
                    dtype='uint8')
    aftB = np.zeros((raw_height, raw_width, 3),
                    dtype='uint8')
    aftRGB = np.zeros((raw_height, raw_width, 3),
                      dtype='uint8')  # 新建和原图长宽一样的RGB图片矩阵

    for r in range(0, raw_height):  # 遍历每个像素
        for c in range(0, raw_width):
            # Red
            aftR[r][c][0] = suppress(rawImage[r][c] * 2 - 255)
            aftRGB[r][c][0] = suppress(rawImage[r][c] * 2 - 255)
            # Green
            aftG[r][c][1] = suppress(255 - rawImage[r][c] * 2)
            aftRGB[r][c][1] = suppress(255 - rawImage[r][c] * 2)
            # Blue
            aftB[r][c][2] = suppress(255 - abs((rawImage[r][c] - 128)*2))
            aftRGB[r][c][2] = suppress(255 - abs((rawImage[r][c] - 128)*2))

    return aftR, aftG, aftB, aftRGB


# 直接执行该文件时运行下列代码
if __name__ == '__main__':
    # 导入图片
    raw = cv2.imread("images/balloon.bmp", 0)  # 0表示以灰度图形式导入，导入后raw是一个二维矩阵

    # 此部分为实际处理
    aftR, aftG, aftB, aftRGB = fakeRGB(raw)

    # 以下代码用于显示图片
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB))
    plt.title("原始图像")

    plt.subplot(2, 3, 2)
    plt.imshow(aftR)
    plt.title("伪彩色增强 R通道")

    plt.subplot(2, 3, 3)
    plt.imshow(aftG)
    plt.title("伪彩色增强 G通道")

    plt.subplot(2, 3, 5)
    plt.imshow(aftB)
    plt.title("伪彩色增强 B通道")

    plt.subplot(2, 3, 6)
    plt.imshow(aftRGB)
    plt.title("伪彩色增强")

    plt.show()
