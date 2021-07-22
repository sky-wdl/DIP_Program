# 用来做代码效果测试的

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = np.array(Image.open('D:/test1.jpg').convert('L'))

plt.figure("lena")

arr = img.flatten()

plt.hist(arr)

plt.show()