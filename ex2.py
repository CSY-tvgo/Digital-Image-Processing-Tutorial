import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ex1


# 实验二：图像的线性变换


# 用于线性变换的函数
# 传入参数： rawImage：原始的灰度图
#           aft_min, aft_max：变换后的灰度范围，可不填，默认为全程线性变换
# 返回值为变换后的图
def LinearTransform(rawImage, aft_min=0, aft_max=255):
    raw_height = rawImage.shape[0]  # 获取原图像的高度
    raw_width = rawImage.shape[1]  # 获取原图像的宽度
    raw_min = rawImage.min()  # 求原图像的最小灰度值
    raw_max = rawImage.max()  # 求原图像的最大灰度值
    aft = np.zeros((raw_height, raw_width), dtype='uint8')  # 新建一个和原图大小一样的全零矩阵
    for r in range(0, raw_height):  # 遍历每个像素
        for c in range(0, raw_width):
            temp = round(aft_min + (rawImage[r][c]-raw_min) *
                         (aft_max-aft_min) / (raw_max-raw_min))  # 对每个像素的灰度进行变换，round指四舍五入
            if temp > 255:  # 防止灰度值超出范围
                temp = 255
            if temp < 0:
                temp = 0
            aft[r][c] = temp  # 把值赋给表示变换后的图像的矩阵
    return aft


# 直接执行该文件时运行下列代码
if __name__ == '__main__':
    # 导入图片
    raw = cv2.imread("./butterfly.bmp", 0)  # 0表示以灰度图形式导入，导入后raw是一个二维矩阵

    # 此部分为实际处理
    raw_hist = ex1.caculateHist(raw)       # 调用实验一中求直方图的函数求原图直方图
    aft1 = LinearTransform(raw)            # 全程线性变换
    aft2 = LinearTransform(raw, 0, 100)  # 变换范围为[100,200]的线性变换
    aft3 = LinearTransform(raw, 100, 200)  # 变换范围为[100,200]的线性变换
    aft4 = LinearTransform(raw, 200, 255)  # 变换范围为[100,200]的线性变换
    aft5 = LinearTransform(raw, 50, 200)  # 变换范围为[100,200]的线性变换

    # 以下代码用于显示图片
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

    plt.figure(figsize=(12, 9), dpi=100)

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB))
    plt.title("原始图像")

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(aft1, cv2.COLOR_GRAY2RGB))
    plt.title("全程线性变换之后")

    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(aft2, cv2.COLOR_GRAY2RGB))
    plt.title("变换范围为[0,100]的线性变换之后")

    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(aft3, cv2.COLOR_GRAY2RGB))
    plt.title("变换范围为[100,200]的线性变换之后")

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(aft4, cv2.COLOR_GRAY2RGB))
    plt.title("变换范围为[200,255]的线性变换之后")

    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(aft5, cv2.COLOR_GRAY2RGB))
    plt.title("变换范围为[50,200]的线性变换之后")

    plt.show()
