import cv2

img = cv2.imread("DemoImages/demo_cone.png")
for i in range(7):
    cv2.imshow("img" + str(i), cv2.cvtColor(img, i))
    print(i)
img = cv2.imread("DemoImages/demo_cone.png", cv2.COLOR_BGR2HLS)
cv2.imshow("img imread mod", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
