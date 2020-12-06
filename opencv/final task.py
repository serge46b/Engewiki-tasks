import cv2

img = cv2.imread('DemoImages/final_challenge.png')
mask1 = cv2.inRange(img, (41, 41, 243), (43, 43, 245))
cv2.imshow("mask1", mask1)
mask2 = cv2.inRange(img, (40, 238, 253), (42, 240, 255))
cv2.imshow("mask2", mask2)

red_dice = cv2.bitwise_and(img, img, mask=mask1)
yellow_dice = cv2.bitwise_and(img, img, mask=mask2)
cv2.imshow("red", red_dice)
cv2.imshow("yellow", yellow_dice)

cv2.waitKey(0)
cv2.destroyAllWindows()
