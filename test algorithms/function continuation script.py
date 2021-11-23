import cv2
import numpy as np
# img = cv2.imread("C:/Engewiki-tasks/duckietown scripts/Data segmentor/road lane line detection/images/img.4.png")
img = np.zeros((480, 640, 3))
[x1, y1] = map(lambda i: int(i), input("enter x1, y1: ").split(", "))
[x2, y2] = map(lambda i: int(i), input("enter x2, y2: ").split(", "))
k = 2
dx = x2 - x1
x3 = x2 + int(dx * k)
y3 = y2 * (k + 1) - y1
print(k, dx, x3, y3)
cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 6)
cv2.line(img, (x1, y1), (x3, y3), (0, 0, 255), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
