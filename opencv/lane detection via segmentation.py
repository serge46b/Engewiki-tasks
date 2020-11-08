import cv2

img = cv2.imread("DemoImages/demo_lane.png")
mask = cv2.inRange(img, (0, 254, 254), (255, 255, 255))
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.imshow("image", img)
cv2.imshow("mask", mask)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
cv2.imshow("contours", img)

cv2.waitKey(0)
cv2.destroyAllWindows()