import cv2
img = cv2.imread("DemoImages/demo_cone.png")
mask = cv2.inRange(img, (30, 52, 100), (89, 131, 255))
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.imshow("img", img)
max_len = 0
idx = 0
for i in range(len(contours)):
    if len(contours[i]) > max_len:
        idx = i
        max_len = len(contours[i])
x, y, w, h = cv2.boundingRect(contours[idx])
box = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.imshow("mask", mask)
cv2.imshow("contours with bounding box", img)
cv2.waitKey(0)
cv2.destroyAllWindows()