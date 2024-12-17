import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def ConvolosiOptimized(f, w):
    # Use OpenCV's filter2D for fast convolution
    g = cv2.filter2D(f, -1, w)
    return g


def KernelBox3x3(f):
    # Define 3x3 box filter (mean filter)
    w = np.ones((3, 3)) / 9.0  # Each element in the kernel is 1/9
    # Apply the optimized Convolosi function using the box filter
    g = ConvolosiOptimized(f, w)
    return g


def GaussianKernel5x5(f):
    # Define 5x5 Gaussian filter
    w = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]]) / 273
    # Apply Gaussian 5x5 kernel
    g = ConvolosiOptimized(f, w)
    return g


def GaussianKernel7x7(f):
    # Define 7x7 Gaussian filter
    w = np.array([[0, 0, 1, 2, 1, 0, 0],
                  [0, 3, 13, 22, 13, 3, 0],
                  [1, 13, 59, 97, 59, 13, 1],
                  [2, 22, 97, 159, 97, 22, 2],
                  [1, 13, 59, 97, 59, 13, 1],
                  [0, 3, 13, 22, 13, 3, 0],
                  [0, 0, 1, 2, 1, 0, 0]]) / 1003
    # Apply Gaussian 7x7 kernel
    g = ConvolosiOptimized(f, w)
    return g


cap = cv2.VideoCapture(0)
current_time = 0
prev_frame_time = 0
filter_applied = False  # Track whether a filter is applied

while True:
    ret, frame = cap.read()  # ret = return, frame = frame

    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        filter_applied = 'box'  # Apply box filter
    elif key == ord('2'):
        filter_applied = 'gaussian5'  # Apply Gaussian 5x5 filter
    elif key == ord('3'):
        filter_applied = 'gaussian7'  # Apply Gaussian 7x7 filter
    elif key == ord('e'):
        filter_applied = None  # Reset to the original image (no filter)
    elif key == ord('4'):
        filter_applied = 'gray'
    elif key == ord('5'):
        filter_applied = 'original'
    elif key == ord('q'):
        # Exit the loop when key 'q' is pressed
        break

    # Apply the selected filter if any
    if filter_applied == 'box':
        output_image = KernelBox3x3(frame)
    elif filter_applied == 'gaussian5':
        output_image = GaussianKernel5x5(frame)
    elif filter_applied == 'gaussian7':
        output_image = GaussianKernel7x7(frame)
    elif filter_applied == 'gray':
        # output_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.destroyAllWindows()
        output_image2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist_original = cv2.calcHist(
            [output_image], [0], None, [256], [0, 256])
        hist_lowpass = cv2.calcHist(
            [output_image2], [0], None, [256], [0, 256])
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.imshow(output_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.plot(hist_original, color='black')
        plt.title('Histogram (Original Image)')
        plt.xlim([0, 256])

        # Plot the low-pass filtered image and its histogram
        plt.subplot(2, 2, 3)
        plt.imshow(output_image2, cmap='gray')
        plt.title('Low-Pass Filtered Image')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.plot(hist_lowpass, color='black')
        plt.title('Histogram (Low-Pass Filtered Image)')
        plt.xlim([0, 256])

        plt.tight_layout()
        plt.show()
    else:
        output_image = frame

    # Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time)
    prev_frame_time = current_time
    fps_text = f'FPS: {fps:.2f}'

    # Display the FPS on the image
    cv2.putText(output_image, fps_text, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the appropriate image (either filtered or original)
    cv2.imshow('Filtered/Normal Image', output_image)


plt.show()


cap.release()
cv2.destroyAllWindows()
