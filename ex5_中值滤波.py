import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import ex3

# 实验五：中值滤波


# 用于中值滤波的函数
# 传入参数： rawImage：原始的灰度图
#           template：模板
# 返回值为变换后的图
def deNoiseByMid(rawImage, template):
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
            temp = raw_expand[r: r+H_height, c: c+H_width] * \
                template  # 对每个像素及其邻点与模板进行点乘
            temp = temp.flatten()  # 把和模板形状一样的temp矩阵展平成一维数组
            temp = np.sort(temp)  # 将temp由小到大排序
            midValue = temp[temp.size -
                            int(temp.nonzero()[0].size / 2) - 1]  # 取非0值的中值
            aft[r][c] = midValue  # 赋给新图

    return aft


# 直接执行该文件时运行下列代码
if __name__ == '__main__':
    # 导入图片
    raw = cv2.imread("images/joker.bmp", 0)  # 0表示以灰度图形式导入，导入后raw是一个二维矩阵

    # 此部分为实际处理
    mask1 = np.array(
        [[1, 1, 1, 1, 1]])
    aft1 = deNoiseByMid(raw, mask1)

    mask2 = np.array(
        [[1],
         [1],
         [1],
         [1],
         [1]])
    aft2 = deNoiseByMid(raw, mask2)

    mask3 = np.array(
        [[0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]])
    aft3 = deNoiseByMid(raw, mask3)

    mask4 = np.array(
        [[1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1]])
    aft4 = deNoiseByMid(raw, mask4)

    mask5 = np.array(
        [[1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1]])/25
    aft5 = ex3.deNoiseByAverage(raw, mask5)

    # 以下代码用于显示图片
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB))
    plt.title("原始图像")

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(aft1, cv2.COLOR_GRAY2RGB))
    plt.title("1*5中值滤波之后")

    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(aft2, cv2.COLOR_GRAY2RGB))
    plt.title("5*1中值滤波之后")

    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(aft3, cv2.COLOR_GRAY2RGB))
    plt.title("5*5十字中值滤波之后")

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(aft4, cv2.COLOR_GRAY2RGB))
    plt.title("5*5方形中值滤波之后")

    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(aft5, cv2.COLOR_GRAY2RGB))
    plt.title("5*5均值滤波之后（用作对比）")

    plt.show()
