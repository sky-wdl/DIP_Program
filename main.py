import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtGui
import maingui
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ------------------选项------------------
# 选择图片文件目录
def click_openImage():
    open_image = tk.Tk
    open_image().withdraw()
    # 定义全局变量用来储存图片文件地址，以方便各个函数调用
    global open_image_path
    # 读取文件路径
    open_image_path = filedialog.askopenfile()
    # 打印目录，调试过程所用，用于排除bug
    print(open_image_path.name)
    # 调用QtGui.QPixmap方法，打开一个图片，存放在变量image中
    image = QtGui.QPixmap(open_image_path.name)
    # 在label里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
    maingui.label.setPixmap(image)


# 退出程序
def click_Exit():
    sys.exit()


# ------------------选项------------------end


# ------------------操作------------------
def click_actionGray_histogram():  # 1
    print('调试输出，证明程序已进入到此函数开始执行，code=1')
    img = Image.open(open_image_path.name)
    Img = img.convert('L')
    Img.save('D:/test1.jpg')
    read_image = cv2.imread('D:/test1.jpg')
    print('code=1.1')
    plt.hist(read_image.ravel(), 256)
    print('code=1.2')
    plt.savefig('D:/test2.jpg')
    print('code=1.3')
    plt.show()
    img = Image.open('D:/test2.jpg')
    img.show()
    # 只要一导入标签2就闪退，问题无解，暂时把下面这句注释掉
#    maingui.label_2.setPixmap('D:/test2.jpg')
    # 开始直方图均衡化，使用equalizeHist函数，导入灰度处理后的图片
    equ = cv2.equalizeHist('D:/test1.jpg')
    # 将两张图片按照水平方式堆叠起来，这样看起来比较有对比
    res = np.hstack((img, equ))
    cv2.imshow('直方图均衡化', res)


def click_actionHistogram_equalization():  # 2
    print('调试输出，证明程序已进入到此函数开始执行，code=2')


def click_actionGray_inversion():  # 3
    print('调试输出，证明程序已进入到此函数开始执行，code=3')


def click_actionLogarithmic_change():  # 4
    print('调试输出，证明程序已进入到此函数开始执行，code=4')


def click_actionImage_plus_noise():  # 5
    print('调试输出，证明程序已进入到此函数开始执行，code=5')


def click_actionSpatial_denoising():  # 6
    print('调试输出，证明程序已进入到此函数开始执行，code=6')


def click_actionFrequency_domain_denoising():  # 7
    print('调试输出，证明程序已进入到此函数开始执行，code=7')


def click_actionEdge_extraction():  # 8
    print('调试输出，证明程序已进入到此函数开始执行，code=8')


# ------------------操作------------------end


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    maingui = maingui.Ui_mainWindow()
    maingui.setupUi(MainWindow)
    # 选项
    maingui.actionOpenFile.triggered.connect(click_openImage)  # 连接信号与槽
    maingui.action12.triggered.connect(click_Exit)  # 连接信号与槽
    # 操作
    maingui.actionGray_histogram.triggered.connect(click_actionGray_histogram)  # 1
    maingui.actionHistogram_equalization.triggered.connect(click_actionHistogram_equalization)  # 2
    maingui.actionGray_inversion.triggered.connect(click_actionGray_inversion)  # 3
    maingui.actionLogarithmic_change.triggered.connect(click_actionLogarithmic_change)  # 4
    maingui.actionImage_plus_noise.triggered.connect(click_actionImage_plus_noise)  # 5
    maingui.actionSpatial_denoising.triggered.connect(click_actionSpatial_denoising)  # 6
    maingui.actionFrequency_domain_denoising.triggered.connect(click_actionFrequency_domain_denoising)  # 7
    maingui.actionEdge_extraction.triggered.connect(click_actionEdge_extraction)  # 8

    MainWindow.show()
    sys.exit(app.exec_())
