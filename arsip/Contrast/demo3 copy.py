import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def ConvolosiOptimized(f, w):
    g = cv2.filter2D(f, -1, w)
    return g


def KernelBox3x3(f):
    w = np.ones((3, 3)) / 9.0
    g = ConvolosiOptimized(f, w)
    return g


def GaussianKernel5x5(f):
    w = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [
                 4, 16, 26, 16, 4], [1, 4, 7, 4, 1]]) / 273
    g = ConvolosiOptimized(f, w)
    return g


def GaussianKernel7x7(f):
    w = np.array([[0, 0, 1, 2, 1, 0, 0], [0, 3, 13, 22, 13, 3, 0], [1, 13, 59, 97, 59, 13, 1], [
                 2, 22, 97, 159, 97, 22, 2], [1, 13, 59, 97, 59, 13, 1], [0, 3, 13, 22, 13, 3, 0], [0, 0, 1, 2, 1, 0, 0]]) / 1003
    g = ConvolosiOptimized(f, w)
    return g


def apply_high_pass_filter(image, kernel):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    output_image = np.zeros_like(image)
    padded_image = np.pad(image, ((pad_height, pad_height),
                          (pad_width, pad_width)), mode='constant', constant_values=0)
    for y in range(image_height):
        for x in range(image_width):
            roi = padded_image[y:y + kernel_height, x:x + kernel_width]
            output_value = np.sum(roi * kernel)
            output_image[y, x] = np.clip(output_value, 0, 255)
    return output_image


def apply_low_pass_filter(image, kernel):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    output_image = np.zeros_like(image)
    padded_image = np.pad(image, ((pad_height, pad_height),
                          (pad_width, pad_width)), mode='constant', constant_values=0)
    for y in range(image_height):
        for x in range(image_width):
            roi = padded_image[y:y + kernel_height, x:x + kernel_width]
            output_value = np.sum(roi * kernel)
            output_image[y, x] = np.clip(output_value, 0, 255)
    return output_image


def apply_sharpen_filter(image, kernel):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    output_image = np.zeros_like(image)
    padded_image = np.pad(image, ((pad_height, pad_height),
                          (pad_width, pad_width)), mode='constant', constant_values=0)
    for y in range(image_height):
        for x in range(image_width):
            roi = padded_image[y:y + kernel_height, x:x + kernel_width]
            output_value = np.sum(roi * kernel)
            output_image[y, x] = np.clip(output_value, 0, 255)
    return output_image


def apply_custom_filter(image, kernel):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    output_image = np.zeros_like(image)
    padded_image = np.pad(image, ((pad_height, pad_height),
                          (pad_width, pad_width)), mode='constant', constant_values=0)
    for y in range(image_height):
        for x in range(image_width):
            roi = padded_image[y:y + kernel_height, x:x + kernel_width]
            output_value = np.sum(roi * kernel)
            output_image[y, x] = np.clip(output_value, 0, 255)
    return output_image

def create_bandreject_filter(image_shape, center_freq, bandwidth):
    rows, cols = image_shape[:2]
    crow, ccol = rows // 2, cols // 2
    
    # Create a mesh grid of coordinates
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u-crow, v-ccol, indexing='ij')
    
    # Calculate the distance from center for each point
    D = np.sqrt((u**2 + v**2))
    
    # Create the band reject filter
    H = 1 / (1 + (bandwidth * D / (D**2 - center_freq**2))**2)
    
    return H

def apply_bandreject_filter(image, center_freq=50, bandwidth=10):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Apply FFT to the image
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    # Step 2: Create the band reject filter
    bandreject_filter = create_bandreject_filter(image.shape, center_freq, bandwidth)
    
    # Step 3: Apply the filter
    filtered_f_shift = f_shift * bandreject_filter
    
    # Step 4: Inverse FFT to get back to spatial domain
    f_ishift = np.fft.ifftshift(filtered_f_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize the output image
    output_image = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return output_image



cap = cv2.VideoCapture(0)
current_time = 0
prev_frame_time = 0
filter_applied = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        filter_applied = 'hpf'
        high_pass_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    elif key == ord('2'):
        filter_applied = 'lpf'
        low_pass_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    elif key == ord('3'):
        filter_applied = 'sharpener'
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    elif key == ord('g'):
        filter_applied = 'gray'
    elif key == ord('4'):
        filter_applied = 'horizontal'
        horizontal_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif key == ord('5'):
        filter_applied = 'vertical'
        vertical_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    elif key == ord('6'):
        filter_applied = 'bandreject'
    elif key == ord('e'):
        filter_applied = None
    elif key == ord('q'):
        break

    if filter_applied == 'hpf':
        output_image = apply_high_pass_filter(frame, high_pass_kernel)
    elif filter_applied == 'lpf':
        output_image = apply_low_pass_filter(frame, low_pass_kernel)
    elif filter_applied == 'sharpener':
        output_image = apply_sharpen_filter(frame, sharpen_kernel)
    elif filter_applied == 'gray':
        output_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_applied == 'horizontal':
        output_image = apply_custom_filter(frame, horizontal_kernel)
    elif filter_applied == 'vertical':
        output_image = apply_custom_filter(frame, vertical_kernel)
    elif filter_applied == 'bandreject':    
        # output_image = apply_bandreject_filter(frame, center_freq=30, bandwidth=5)
        output_image = apply_bandreject_filter(frame)
        

    else:
        output_image = frame

    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time)
    prev_frame_time = current_time
    fps_text = f'FPS: {fps:.2f}'
    cv2.putText(output_image, fps_text, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Filtered/Normal Image', output_image)

cap.release()
cv2.destroyAllWindows()
