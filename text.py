import cv2
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('D:/image/3.png')
# cv2.imshow("src", src)
a = src.ravel()
plt.hist(a, 256)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()