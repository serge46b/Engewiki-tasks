import numpy as np
import cv2
img = cv2.imread('DemoImages/stop_template.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('DemoImages/demo_stop.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp, des = sift.detectAndCompute(gray, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
#img = cv2.drawKeypoints(gray, kp, img)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des, des2, k=2)
good = [[m] for m, n in matches if m.distance < 0.7*n.distance]
img3 = cv2.drawMatchesKnn(img, kp, img2, kp2, good, None)

cv2.imshow('matches', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
