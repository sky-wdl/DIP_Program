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


# ------------------公共函数------------------
# 返回尺寸
def resize(w, h, w_box, h_box, pil_image):
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)


# 显示图片
def showimg(img, isgray=False):
    plt.axis("off")
    if isgray == True:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()


# ------------------公共函数------------------end


# ------------------全局变量------------------
# ------------------全局变量------------------end


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
    #  src = cv2.imread(open_image_path.name)
    #  cv2.imshow("src", src)
    #
    #  plt.hist(src.ravel(), 256)
    #  plt.show()
    #
    #  cv2.waitKey(0)
    #  cv2.destroyAllWindows()
    #
    #  img_l = img.convert('L')
    #  img_l.save('D:/test1.jpg')
    #  img_l.show()
    #  read_img = cv2.imread('D:/test1.jpg')
    #  print('code=1.0')
    #  img_l = cv2.imread('D:/test1.jpg')
    #
    #  print('code=1.1')
    #  img_ravel = img_l.ravel()
    #
    #  print('code=1.2')
    #  plt.hist(img_ravel, 256)
    #
    #  print('code=1.3')
    #  plt.savefig('D:/test2.jpg')
    # print('code=1.3')
    # plt.show()
    # img_l.show()
    #  只要一导入标签2就闪退，问题无解，暂时把下面这句注释掉
    # maingui.label_2.setPixmap('D:/test2.jpg')
    #
    #  开始直方图均衡化，使用equalizeHist函数，导入灰度处理后的图片
    # equ = cv2.equalizeHist(open_image_path.name)
    #  将两张图片按照水平方式堆叠起来，这样看起来比较有对比
    # res = np.hstack((src, equ))
    # cv2.imshow('直方图均衡化', res)
    img = Image.open(open_image_path.name)
    print('即将进行灰度变化，code=1.0')
    src = np.array(img.convert("L"))
    print('灰度变化end，进行矩阵设置，code=1.1')
    dest = np.zeros_like(src)
    print('矩阵设置end，进行直方图，code=1.2')

    src1_for_hist = np.array(src).reshape(1, -1).tolist()
    plt.title("灰度直方图", fontsize='16')
    plt.hist(src1_for_hist, bins=255, density=0)
    plt.show()
    print('进行直方图end，code=1.3')

    src2_for_hist = np.array(dest).reshape(1, -1).tolist()
    plt.title("灰度直方图(均衡化)", fontsize='16')
    plt.hist(src2_for_hist, bins=255, density=0)
    plt.show()
    print('进行灰度直方图end，code=1.4')

    plt.title("灰度图", fontsize='16')
    plt.imshow(src, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()
    print('进行灰度图end，code=1.5')

    gray_times = []
    for i in range(256):
        gray_times.append(np.sum(src == i))
    normalied_gray = gray_times / np.sum(gray_times)
    rk2sk = []
    for rk in range(256):
        sk = 0
        for k in range(rk):
            sk += normalied_gray[k]
        rk2sk.append(256 * sk)

    width, height = src.shape
    for i in range(width - 1):
        for j in range(height):
            dest[i][j] = rk2sk[src[i][j]]

    plt.title("灰度图（均衡化）", fontsize='16')
    plt.imshow(dest, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()
    print('进行灰度图均衡化end，code=1.6')


def click_actionHistogram_equalization():  # 2灰度变化
    print('调试输出，证明程序已进入到此函数开始执行，code=2')
    # 灰度反相
    im = Image.open(open_image_path.name)
    im_l = im.convert('L')
    im_l.show()
    # im_arr = np.array(im_gray)
    # im1 = 255 - im_arr
    # showimg(Image.fromarray(im1))

    print('code=2.0')
    img_gray = im_l.point(lambda i: 256 - i - 1)
    plt.title("灰度反转", fontsize='16')
    plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    print('code=2.1')
    img_log = im_l.point(lambda i: 20 * 255 * np.log(1 + i / 255))
    plt.title("对数变换", fontsize='16')
    plt.imshow(img_log, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    print('code=2.2')
    imarray = np.array(im_l)
    height, width = imarray.shape
    for i in range(height):
        for j in range(width):
            aft = int(imarray[i, j] + 55)
            if aft <= 255 and aft >= 0:
                imarray[i, j] = aft
            elif aft > 255:
                imarray[i, j] = 255
            else:
                imarray[i, j] = 0
    img_lin = Image.fromarray(imarray)
    plt.title("线性变换", fontsize='16')
    plt.imshow(img_lin, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()
    print('code=2,end')


def click_actionGray_inversion():  # 3几何变换
    print('调试输出，证明程序已进入到此函数开始执行，code=3')
    img = Image.open(open_image_path.name)
    # 获取图像的大小
    print(img.size)
    # 获取图像 width
    print(img.size[0])
    # 获取图像 height
    print(img.size[1])

    img_2 = img.resize((img.size[0] * 2, img.size[1] * 2), Image.ANTIALIAS)  # 放大两倍
    img.show()
    img_2.show()
    print('code=3.1,end')

    img_3 = img.resize((int(img.size[0] * 0.5), int(img.size[1] * 0.5)), Image.ANTIALIAS)  # 缩小两倍
    img_3.show()
    print('code=3.2,end')

    # mg = img.transpose(Image.ROTATE_90)  # 将图片旋转90度
    img_4 = img.transpose(Image.ROTATE_180)  # 将图片旋转180度
    # img = img.transpose(Image.ROTATE_270)  # 将图片旋转270度
    img_4.show()
    # img.save("img/rotateImg.png")
    print('code=3.3,end')

    img = cv2.imread(open_image_path.name)
    rows, cols, _ = img.shape
    M = np.float32([[1, 0, -100], [0, 1, 50]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow("2", dst)
    print('code=3.4,end')


def click_actionLogarithmic_change():  # 4
    print('调试输出，证明程序已进入到此函数开始执行，code=4')


def click_actionImage_plus_noise():  # 5
    print('调试输出，证明程序已进入到此函数开始执行，code=5')



def click_actionSpatial_denoising():  # 6
    print('调试输出，证明程序已进入到此函数开始执行，code=6')
    # 拉普拉斯变换
    laplace = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])

    laplace2 = np.array([[1, 1, 1],
                         [1, -8, 1],
                         [1, 1, 1]])

    sobel_filter_v = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1], ])

    sobel_filter_h = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1], ])


def click_actionFrequency_domain_denoising():  # 7
    print('调试输出，证明程序已进入到此函数开始执行，code=7')


def click_actionEdge_extraction():  # 8
    print('调试输出，证明程序已进入到此函数开始执行，code=8')
    # robert 算子
    img = cv2.imread(open_image_path.name, cv2.IMREAD_GRAYSCALE)
    r, c = img.shape
    r_sunnzi = [[-1, -1], [1, 1]]
    for x in range(r):
        for y in range(c):
            if (y + 2 <= c) and (x + 2 <= r):
                imgChild = img[x:x + 2, y:y + 2]
                list_robert = r_sunnzi * imgChild
                img[x, y] = abs(list_robert.sum())
    plt.title("robert算子", fontsize='16')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    # sobel算子
    img = cv2.cvtColor(np.array(open_image_path.name), cv2.COLOR_BGR2GRAY)
    r, c = img.shape
    new_image = np.zeros((r, c))
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    s_suanziX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # X方向
    s_suanziY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    for i in range(r - 2):
        for j in range(c - 2):
            new_imageX[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziX))
            new_imageY[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziY))
            new_image[i + 1, j + 1] = (new_imageX[i + 1, j + 1] * new_imageX[i + 1, j + 1] + new_imageY[i + 1, j + 1] *
                                       new_imageY[i + 1, j + 1]) ** 0.5
    img_1 = np.uint8(new_image)
    plt.title("sobel算子", fontsize='16')
    plt.imshow(img_1, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    # Laplace算子
    r, c = img.shape
    new_image = np.zeros((r, c))
    L_sunnzi = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    for i in range(r - 2):
        for j in range(c - 2):
            new_image[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * L_sunnzi))
    img_2 = np.uint8(new_image)
    plt.title("laplace算子", fontsize='16')
    plt.imshow(img_2, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()


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
