import cv2
img = cv2.imread("DemoImages/demo_cone.png")
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()