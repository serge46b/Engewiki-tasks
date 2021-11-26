import numpy as np

[x1, y1] = map(lambda i: int(i), input("enter x1, y1: ").split(", "))
[x2, y2] = map(lambda i: int(i), input("enter x2, y2: ").split(", "))
[x3, y3] = map(lambda i: int(i), input("enter x2, y2: ").split(", "))

a = np.radians(np.array([x1, y1]))
b = np.radians(np.array([x2, y2]))
c = np.radians(np.array([x3, y3]))

avec = a - b
cvec = c - b

lat = b[0]
avec[1] *= np.cos(lat)
cvec[1] *= np.cos(lat)

deg = np.degrees(np.arccos(np.dot(avec, cvec) / (np.linalg.norm(avec) * np.linalg.norm(cvec))))
print(deg)
