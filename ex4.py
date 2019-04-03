import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# 实验四：锐化


# 使用拉普拉斯算子进行锐化的函数
# 传入参数： rawImage：原始的灰度图
#           alpha: 锐化程度
# 返回值为变换后的图
def sharpen_Laplacian(rawImage, alpha):
    mask = np.array(
        [[0, -1, 0],
         [-1, 4, -1],
         [0, -1, 0]])  # 拉普拉斯锐化模板
    H_height = 3
    H_width = 3

    raw_height = rawImage.shape[0]  # 获取原图像的高度
    raw_width = rawImage.shape[1]  # 获取原图像的宽度

    # 把原图四周拓展一圈
    raw_expand = rawImage
    expansion = raw_expand[0: int(H_height/2), 0: raw_width]
    raw_expand = np.r_[expansion, raw_expand]
    expansion = raw_expand[raw_height: raw_height + int(H_height/2),
                           0: raw_width]
    raw_expand = np.r_[raw_expand, expansion]
    expansion = raw_expand[0: raw_height + H_height - 1, 0: int(H_width/2)]
    raw_expand = np.c_[expansion, raw_expand]
    expansion = raw_expand[0: raw_height + H_height - 1,
                           raw_width: raw_width+int(H_width/2)]
    raw_expand = np.c_[raw_expand, expansion]

    aft = np.zeros((raw_height, raw_width), dtype='uint8')  # 新建一个和原图大小一样的全零矩阵

    for r in range(0, raw_height):  # 遍历每个像素
        for c in range(0, raw_width):
            temp1 = 0
            for i in range(0, 3):  # 将该像素周围的灰度与模板中对应的值相乘然后相加
                for j in range(0, 3):
                    temp1 = temp1 + mask[i][j] * raw_expand[r + i][c + j]
            temp2 = rawImage[r][c] + alpha * temp1  # 把值赋给表示变换后的图像的矩阵
            if temp2 > 255:  # 防止灰度值超出范围
                temp2 = 255
            if temp2 < 0:
                temp2 = 0
            aft[r][c] = temp2

    return aft


# 直接执行该文件时运行下列代码
if __name__ == '__main__':
    # 导入图片
    raw = cv2.imread("./build.bmp", 0)  # 0表示以灰度图形式导入，导入后raw是一个二维矩阵

    # 此部分为实际处理
    aft1 = sharpen_Laplacian(raw, 0.25)
    aft2 = sharpen_Laplacian(raw, 0.5)
    aft3 = sharpen_Laplacian(raw, 1)
    aft4 = sharpen_Laplacian(raw, 2)
    aft5 = sharpen_Laplacian(raw, 4)

    # 以下代码用于显示图片
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB))
    plt.title("原始图像")

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(aft1, cv2.COLOR_GRAY2RGB))
    plt.title("锐化强度0.25")

    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(aft2, cv2.COLOR_GRAY2RGB))
    plt.title("锐化强度0.5")

    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(aft3, cv2.COLOR_GRAY2RGB))
    plt.title("锐化强度1")

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(aft4, cv2.COLOR_GRAY2RGB))
    plt.title("锐化强度2")

    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(aft5, cv2.COLOR_GRAY2RGB))
    plt.title("锐化强度4")

    plt.show()
