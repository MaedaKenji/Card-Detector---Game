import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar
# 0 untuk membaca sebagai grayscale
img = cv2.imread('contrast/itsSurabaya2.jpg', 0)

# Plot histogram menggunakan plt.hist()
plt.figure(figsize=(10, 6))
plt.hist(img.ravel(), bins=256, range=[0, 256])
plt.title('Histogram Intensitas Piksel')
plt.xlabel('Intensitas Piksel')
plt.ylabel('Jumlah Piksel')
plt.xlim([0, 256])
plt.grid(True)
plt.show()

# Tampilkan gambar asli
plt.figure(figsize=(8, 6))
plt.imshow(img, cmap='gray')
plt.title('Gambar Asli')
plt.axis('off')
plt.show()
