import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# 实验七：边缘检测


# 使用拉普拉斯算子进行边缘检测的函数
# 传入参数： rawImage：原始的灰度图
#           threshold: 阈值，默认为32
# 返回值为变换后的图
def edge_Laplacian(rawImage, threshold=-1):
    if threshold == -1:
        threshold = 32

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
            if temp1 >= threshold:
                aft[r][c] = 0
            else:
                aft[r][c] = 255

    return aft


# 直接执行该文件时运行下列代码
if __name__ == '__main__':
    # 导入图片
    raw = cv2.imread("./train.bmp", 0)  # 0表示以灰度图形式导入，导入后raw是一个二维矩阵

    # 此部分为实际处理
    th1, th2, th3, th4, th5 = 16, 32, 48, 64, 80
    aft1 = edge_Laplacian(raw, th1)
    aft2 = edge_Laplacian(raw, th2)
    aft3 = edge_Laplacian(raw, th3)
    aft4 = edge_Laplacian(raw, th4)
    aft5 = edge_Laplacian(raw, th5)

    # 以下代码用于显示图片
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB))
    plt.title("原始图像")

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(aft1, cv2.COLOR_GRAY2RGB))
    plt.title("拉普拉斯算子边缘检测 阈值"+str(th1))

    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(aft2, cv2.COLOR_GRAY2RGB))
    plt.title("拉普拉斯算子边缘检测 阈值"+str(th2))

    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(aft3, cv2.COLOR_GRAY2RGB))
    plt.title("拉普拉斯算子边缘检测 阈值"+str(th3))

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(aft4, cv2.COLOR_GRAY2RGB))
    plt.title("拉普拉斯算子边缘检测 阈值"+str(th4))

    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(aft5, cv2.COLOR_GRAY2RGB))
    plt.title("拉普拉斯算子边缘检测 阈值"+str(th5))

    plt.show()
