import cv2
import numpy as np
from cv2 import aruco
from glob import glob
from PIL import Image

# loading images
image_sequence = []
for img_path in glob("imgs_old/*.png"):
    image_sequence.append(Image.open(img_path).convert("RGB"))

# aruco init
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
arucoParams = aruco.DetectorParameters_create()

# process images
markers_data = {}
for img_idx in range(len(image_sequence)):
    markers_data[img_idx] = {}
    img = cv2.cvtColor(np.array(image_sequence[img_idx]), cv2.COLOR_RGB2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)
    if len(corners) == 0:
        markers_data[img_idx]["corners"] = []
        markers_data[img_idx]["ids"] = []
        markers_data[img_idx]["center"] = []
        continue
    markers_data[img_idx]["corners"] = corners
    markers_data[img_idx]["ids"] = ids
    centers = []
    for c in corners:
        m = cv2.moments(c)
        cx = m['m10'] / (m['m00'] + 1e-5)
        cy = m['m01'] / (m['m00'] + 1e-5)
        centers.append((cx, cy))
    markers_data[img_idx]["centers"] = centers


# visualize processed points
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


for img_idx in range(len(image_sequence)):
    corners = markers_data[img_idx]["corners"]
    if len(corners) == 0:
        continue
    ids = markers_data[img_idx]["ids"]
    centers = markers_data[img_idx]["centers"]

    img = np.array(image_sequence[img_idx])
    img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 0))
    for c in centers:
        cv2.circle(img, (int(c[0]), int(c[1])), 2, (0, 0, 255), -1)
    cv2.imshow("tags", resize_with_aspect_ratio(img, height=700))
    q = cv2.waitKey(0)
    if q == 27:
        break

cv2.destroyAllWindows()




