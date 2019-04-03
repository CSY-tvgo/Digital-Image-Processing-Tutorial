import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ex1


# 实验二_额外：图像的直方图均衡化


# 用于直方图均衡化的函数，传入参数为原始的灰度图，返回值为变换后的图
def HistEqualize(rawImage):
    raw_height = rawImage.shape[0]  # 获取原图像的高度
    raw_width = rawImage.shape[1]  # 获取原图像的宽度

    hist = ex1.caculateHist(rawImage)  # 调用实验一中写的的函数求各个灰度级的频率

    prob_sum = np.zeros(256)  # 初始化一个长度为256的全零向量，用以存储累积概率
    prob_sum[0] = hist[0]  # 计算累积概率
    for i in range(1, 256):
        prob_sum[i] = prob_sum[i-1] + hist[i]

    T = np.zeros(256, dtype='uint8')  # 初始化一个长度为256的全零向量，用以存储各个灰度经过变换后的新灰度
    for i in range(0, 256):
        T[i] = round(prob_sum[i] * 255)  # 计算变换前后各个灰度的映射关系

    aft = np.zeros((raw_height, raw_width), dtype='uint8')  # 新建一个和原图大小一样的全零矩阵
    for r in range(0, raw_height):  # 遍历每个像素
        for c in range(0, raw_width):
            aft[r][c] = T[rawImage[r][c]]  # 对每个像素的灰度进行变换，round指四舍五入

    return aft


# 直接执行该文件时运行下列代码
if __name__ == '__main__':
    # 导入图片
    raw = cv2.imread("images/butterfly.bmp", 0)  # 0表示以灰度图形式导入，导入后raw是一个二维矩阵

    # 此部分为实际处理
    aft1 = HistEqualize(raw)            # 直方图均衡化

    # 以下代码用于显示图片
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

    plt.figure(figsize=(12, 9), dpi=100)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB))
    plt.title("原始图像")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(aft1, cv2.COLOR_GRAY2RGB))
    plt.title("直方图均衡化之后")

    plt.show()
