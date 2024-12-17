import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


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


def create_histogram(img):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(img.ravel(), bins=256, range=[0, 256], color='b', alpha=0.7)
    ax.set_title('Histogram')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    hist_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    hist_img = hist_img.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return hist_img


# Inisialisasi video capture
cap = cv2.VideoCapture(0)  # 0 untuk webcam default

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Terapkan contrast stretching
    r_min, r_max = np.min(gray), np.max(gray)  # r_min
    i_min, i_max = 80, 120
    r1, s1, r2, s2 = 80, 20, 175, 240
    # stretched = contrast_stretching(gray, r_min, r_max, i_min, i_max)
    stretched = contrast_stretching(gray, r1, s1, r2, s2)

    # Buat histogram
    hist_original = create_histogram(gray)
    hist_stretched = create_histogram(stretched)

    # Tampilkan hasil
    cv2.imshow('Original', gray)
    cv2.imshow('Contrast Stretched', stretched)
    cv2.imshow('Histogram Original', hist_original)
    cv2.imshow('Histogram Stretched', hist_stretched)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
