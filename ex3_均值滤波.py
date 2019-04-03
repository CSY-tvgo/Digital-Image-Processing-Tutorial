import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# 实验三：邻域平均消噪处理


# 用于邻域平均消噪的函数
# 传入参数： rawImage：原始的灰度图
#           template：模板
#           threshold：阈值，默认为0
# 返回值为变换后的图
def deNoiseByAverage(rawImage, template, threshold=0):
    raw_height = rawImage.shape[0]  # 获取原图像的高度
    raw_width = rawImage.shape[1]  # 获取原图像的宽度
    H_height = template.shape[0]  # 获取模板的高度
    H_width = template.shape[1]  # 获取模板的宽度
    if (H_height % 2 == 0) or (H_width % 2 == 0):
        exit("\n错误，模板的长宽应当为奇数！\n")
        return

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
            temp = 0
            for i in range(0, H_height):  # 将该像素周围的灰度与模板中对应的值相乘然后相加
                for j in range(0, H_width):
                    temp = temp + template[i][j] * raw_expand[r + i][c + j]
            if temp > 255:  # 防止灰度值超出范围
                temp = 255
            if temp < 0:
                temp = 0
            if abs(rawImage[r][c]-temp) > threshold:
                aft[r][c] = temp  # 把值赋给表示变换后的图像的矩阵
            else:
                aft[r][c] = rawImage[r][c]
    return aft


# 直接执行该文件时运行下列代码
if __name__ == '__main__':
    # 导入图片
    raw = cv2.imread("images/bird.bmp", 0)  # 0表示以灰度图形式导入，导入后raw是一个二维矩阵

    # 此部分为实际处理
    H0 = np.array(
        [[1/9, 1/9, 1/9],
         [1/9, 1/9, 1/9],
         [1/9, 1/9, 1/9]])  # 均值模板
    aft1 = deNoiseByAverage(raw, H0)

    H2 = np.array(
        [[1/16, 2/16, 1/16],
         [2/16, 4/16, 2/16],
         [1/16, 2/16, 1/16]])  # 高斯模板
    aft2 = deNoiseByAverage(raw, H2)

    HXX = np.array(
        [[1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
         [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
         [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
         [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
         [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
         [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
         [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49]])  # 随便玩玩的模板
    th3 = 0  # XXX:在这里改阈值
    th4 = 10
    th5 = 20
    aft3 = deNoiseByAverage(raw, HXX, th3)
    aft4 = deNoiseByAverage(raw, HXX, th4)
    aft5 = deNoiseByAverage(raw, HXX, th5)

    # 以下代码用于显示图片
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB))
    plt.title("原始图像")

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(aft1, cv2.COLOR_GRAY2RGB))
    plt.title("3*3均值模板消噪之后")

    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(aft2, cv2.COLOR_GRAY2RGB))
    plt.title("3*3高斯模板消噪之后")

    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(aft3, cv2.COLOR_GRAY2RGB))
    plt.title("7*7均值模板 阈值" + str(th3))

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(aft4, cv2.COLOR_GRAY2RGB))
    plt.title("7*7均值模板 阈值" + str(th4))

    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(aft5, cv2.COLOR_GRAY2RGB))
    plt.title("7*7均值模板 阈值" + str(th5))

    plt.show()
