import cv2
import numpy as np
import matplotlib.pyplot as plt


def contrast_stretching(img, r1, s1, r2, s2):
    # Ensure input image is float
    img = img.astype(float)

    # Apply the T function to each pixel in the image
    T_applied = np.vectorize(lambda r: T(r, r1, s1, r2, s2))

    # Apply T to the entire image
    stretched = T_applied(img)

    # Clip values to ensure they are within the valid range
    stretched = np.clip(stretched, 0, 255)

    # Convert back to uint8
    return stretched.astype(np.uint8)


def T(r, r1, s1, r2, s2):
    s = 0
    if (0 < r) & (r < r1):
        s = s1 / r1 * r
    elif (r1 <= r) & (r < r2):
        s = (s2 - s1) / (r2 - r1) * (r - r1) + s1
    elif (r2 <= r) & (r <= 255):
        s = (255 - s2) / (255 - r2) * (r - r2) + s2
    else:
        s = s2
    s = np.uint8(np.floor(s))
    return s


def apply_transformation(img, r1, s1, r2, s2):
    transformed_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            transformed_img[i, j] = T(img[i, j], r1, s1, r2, s2)
    return transformed_img


# Baca gambar
# 0 untuk membaca sebagai grayscale
img = cv2.imread('Contrast/itsSurabaya2.jpg', 0)

# Terapkan contrast stretching
# Get min and max pixel values from the image
r_min, r_max = np.min(img), np.max(img)

# Set desired output range (usually 0-255 for 8-bit images)
i_min, i_max = 80, 120

r1, s1, r2, s2 = 80, 20, 175, 240

# stretched_img = contrast_stretching(img, r_min, r_max, i_min, i_max)
stretched_img = contrast_stretching(img, r1, s1, r2, s2)
# stretched_img = T(r_min, r_min, i_min, r_max, i_max)
# stretched_img = apply_transformation(img, r_min, i_min, r_max, i_max)

# Tampilkan gambar asli dan hasil filter menggunakan cv2
cv2.imshow('Original Image', img)
cv2.imshow('Contrast Stretched Image', stretched_img)


# Plot histogram untuk gambar asli dan hasil filter
plt.figure(figsize=(12, 6))

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(222)
plt.hist(img.ravel(), bins=256, range=[0, 256], color='b', alpha=0.7)
plt.title('Histogram of Original Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(223)
plt.imshow(stretched_img, cmap='gray')
plt.title('Contrast Stretched Image')
plt.axis('off')

plt.subplot(224)
plt.hist(stretched_img.ravel(), bins=256, range=[0, 256], color='r', alpha=0.7)
plt.title('Histogram of Stretched Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
