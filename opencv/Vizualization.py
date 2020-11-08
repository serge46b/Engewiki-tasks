import cv2
import numpy as np
img = np.zeros((512, 512, 3))
cv2.line(img, (0, 0), (512, 512), (255, 255, 255), 5)
cv2.circle(img, (256, 256), 50, (0,255,255), 3)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()