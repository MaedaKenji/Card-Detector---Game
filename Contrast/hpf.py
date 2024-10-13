import cv2
import numpy as np


def Convolosi(f, w):
    f = np.float64(f)/255
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
    return g


# Membaca File Citra
f = cv2.imread("Contrast\itsSurabaya2.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow('Citra Asli', f)

w = np.array([[0, 1, 0], [1, -4., 1], [0, 1, 0]])
g = Convolosi(f, w)
cv2.imshow('Highpass Filter', g)
cv2.waitKey(0)
cv2.destroyAllWindows()
