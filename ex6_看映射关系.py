import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# 该程序用于查看ex6.py中伪彩色增强的映射关系

# 把值抑制在0~255内
def suppress(num):
    if num > 255:
        return 255
    elif num < 0:
        return 0
    else:
        return num

# 直接执行该文件时运行下列代码
if __name__ == '__main__': 
    a = np.zeros((100, 255, 3), dtype='uint8')
    for i in range(0, 50):
        for j in range(0, 255):
            a[i, j, 0] = j
            a[i, j, 1] = j
            a[i, j, 2] = j
    for r in range(50, 100):  # 遍历每个像素
        for c in range(0, 255):
            # Red
            a[r][c][0] = suppress(c * 2 - 255) 
            # Green
            a[r][c][1] = suppress(255 - c * 2) 
            # Blue
            a[r][c][2] = suppress(255 - abs((c - 128)*2)) 

    # 以下代码用于显示图片
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

    plt.subplot(1, 1, 1)
    plt.imshow(a)
    plt.title("ex6_伪彩色增强的映射关系")

    plt.show()
