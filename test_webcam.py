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
    cam = cv2.VideoCapture(1)
    while cv2.waitKey(1) & 0xFF != ord('q'):
        ret, raw = cam.read()
        raw = cv2.resize(raw, (100, 75))
        aft = HistEqualize_RGB(raw)
        aft = cv2.resize(aft, (2000, 1500))
        cv2.imshow('HistEqualized', aft)
    cam.release()
    cv2.destroyAllWindows()
