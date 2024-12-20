import cv2
from matplotlib import pyplot as plt
import numpy as np


def Convolosi(f, w):
    baris = f.shape[0]
    kolom = f.shape[1]
    dkernel = w.shape[0]
    dkernel2 = np.int32(np.floor(dkernel/2))

    g = np.zeros((baris, kolom))
    for y in range(baris):
        for x in range(kolom):
            g[y, x] = 0
            for i in range(dkernel):
                yy = y+i-dkernel2
                if (yy < 0) | (yy >= baris-1):
                    continue
                for j in range(dkernel):
                    xx = x+j - dkernel2
                    if (xx < 0) | (xx >= kolom-1):
                        continue
                    g[y, x] = g[y, x]+f[yy, xx]*w[i, j]
                # end for
            # end for
        # end for
    # end for
    g = np.uint8(np.floor(g))
    return g


# Membaca File Citra
f = cv2.imread("Contrast\itsSurabaya2.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow('Citra Asli', f)

w = np.array([[0.3679, 0.6065, 0.3679], [0.6065, 1.0000, 0.6065],
             [0.3679, 0.6065, 0.3679]])/4.8976
g3 = Convolosi(f, w)
cv2.imshow('Hasil Gaussian Kernel 3x3', g3)


w = np.array([
    [1, 4, 7, 4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4, 7, 4, 1]])/273

g5 = Convolosi(f, w)
cv2.imshow('Hasil Gaussian Kernel 5x5', g5)

w = np.array([[0, 0, 1, 2, 1, 0, 0],
              [0, 3, 13, 22, 13, 3, 0],
              [1, 13, 59, 97, 59, 13, 1],
              [2, 22, 97, 159, 97, 22, 2],
              [1, 13, 59, 97, 59, 13, 1],
              [0, 3, 13, 22, 13, 3, 0],
              [0, 0, 1, 2, 1, 0, 0]])/1003

g7 = Convolosi(f, w)
cv2.imshow('Hasil Gaussian Kernel 7x7', g7)

cv2.waitKey(0)
cv2.destroyAllWindows()
