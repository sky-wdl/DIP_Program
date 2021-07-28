# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from PIL import Image,ImageFilter
import matplotlib.pylab as plt
from matplotlib import pyplot as plt



def MyLaplace(img, window, c):
   m = window.shape[0]
   n = window.shape[1]
   
   img_border = np.zeros((img.shape[0]+ m - 1, img.shape[1] + n -1))
   img_border[(m - 1)//2:(img.shape[0] + (m - 1)// 2),
              (n - 1)//2:(img.shape[1] + (n - 1)// 2)] = img
   
   img_result = np.zeros(img.shape)
   for i in range(img.shape[0]):
         for j in range(img.shape[1]):
             temp = img_border[i : i + m, j : j + n]
             img_result[i,j] = np.sum(np.multiply(temp, window))
   img_result = img + c*img_result
   return Image.fromarray(img_border), Image.fromarray(img_result)

laplace = np.array([[0,1,0],
                    [1,-4,1],
                    [0,1,0]])

laplace2 = np.array([[1,1,1],
                    [1,-8,1],
                    [1,1,1]])
         
sobel_filter_v = np.array([[-1,0,1],
                           [-2,0,2],
                           [-1,0,1],])

sobel_filter_h = np.array([[-1,-2,-1],
                           [0,0,0],
                           [1,2,1],])
 
img = np.array(Image.open("D:/images/lena.jpg").convert("L"))
img_border,img_result =MyLaplace(img, sobel_filter_h, 0.2)
img_border.show()
img_result.show()