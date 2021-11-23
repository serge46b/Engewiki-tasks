"""IsLinesCross(_int64 x11, _int64 y11, _int64 x12, _int64 y12, _int64 x21, _int64 y21, _int64 x22, _int64 y22)
{

_int64 maxx1 = max(x11, x12), maxy1 = max(y11, y12);
_int64 minx1 = min(x11, x12), miny1 = min(y11, y12);
_int64 maxx2 = max(x21, x22), maxy2 = max(y21, y22);
_int64 minx2 = min(x21, x22), miny2 = min(y21, y22);

if (minx1 > maxx2 || maxx1 < minx2 || miny1 > maxy2 || maxy1 < miny2)
  return FALSE;  // Момент, када линии имеют одну общую вершину...


_int64 dx1 = x12-x11, dy1 = y12-y11; // Длина проекций первой линии на ось x и y
_int64 dx2 = x22-x21, dy2 = y22-y21; // Длина проекций второй линии на ось x и y
_int64 dxx = x11-x21, dyy = y11-y21;
_int64 div, mul;


if ((div = (_int64)((double)dy2*dx1-(double)dx2*dy1)) == 0)
  return FALSE; // Линии параллельны...
if (div > 0) {
  if ((mul = (_int64)((double)dx1*dyy-(double)dy1*dxx)) < 0 || mul > div)
    return FALSE; // Первый отрезок пересекается за своими границами...
  if ((mul = (_int64)((double)dx2*dyy-(double)dy2*dxx)) < 0 || mul > div)
     return FALSE; // Второй отрезок пересекается за своими границами...
}

if ((mul = -(_int64)((double)dx1*dyy-(double)dy1*dxx)) < 0 || mul > -div)
  return FALSE; // Первый отрезок пересекается за своими границами...
if ((mul = -(_int64)((double)dx2*dyy-(double)dy2*dxx)) < 0 || mul > -div)
  return FALSE; // Второй отрезок пересекается за своими границами...

return TRUE;
}"""

import cv2
import numpy as np

# img = cv2.imread("C:/Engewiki-tasks/duckietown scripts/Data segmentor/road lane line detection/images/img.4.png")
img = np.zeros((480, 640, 3))
[x11, y11] = map(lambda i: int(i), input("enter x11, y11: ").split(", "))
[x12, y12] = map(lambda i: int(i), input("enter x12, y12: ").split(", "))
[x21, y21] = map(lambda i: int(i), input("enter x21, y21: ").split(", "))
[x22, y22] = map(lambda i: int(i), input("enter x22, y22: ").split(", "))

maxx1 = max(x11, x12)
maxy1 = max(y11, y12)
minx1 = min(x11, x12)
miny1 = min(y11, y12)
maxx2 = max(x21, x22)
maxy2 = max(y21, y22)
minx2 = min(x21, x22)
miny2 = min(y21, y22)

if minx1 > maxx2 or maxx1 < minx2 or miny1 > maxy2 or maxy1 < miny2:
  print("no intersect")
else:
    dx1 = x12-x11
    dy1 = y12-y11
    dx2 = x22-x21
    dy2 = y22-y21
    dxx = x11-x21
    dyy = y11-y21
    div = int(dy2*dx1-dx2*dy1)
    mul = 0
    if div == 0:
        print("no intersect 1")
    else:
        if div > 0:
            mul = int(dx1*dyy-dy1*dxx)
            if mul < 0 or mul > div:
                print("no intersect 2")
            else:
                mul = int(dx2*dyy-dy2*dxx)
                if mul < 0 or mul > div:
                    print("no intersect 3")
                else:
                    print('intersect')
        else:
            mul = -int(dx1*dyy-dy1*dxx)
            if mul < 0 or mul > -div:
                print("no intersect 4")
            else:
                mul = -int(dx2*dyy-dy2*dxx)
                if mul < 0 or mul > -div:
                    print("no intersect 5")
                else:
                    print('intersect')
cv2.line(img, (x11, y11), (x12, y12), (0, 255, 0), 2)
cv2.line(img, (x21, y21), (x22, y22), (0, 0, 255), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
