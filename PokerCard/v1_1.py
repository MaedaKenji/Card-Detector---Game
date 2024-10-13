import cv2
import numpy as np

# Step 1: Load the image
image = cv2.imread('poker.jpg')

# Step 2: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 4: Perform edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Step 5: Find contours in the image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Loop over contours and approximate the largest one as a polygon
for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # If the approximated contour has four points, it's likely a card
    if len(approx) == 4:
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)  # Draw green contour around the card

# Step 7: Display the result
cv2.imshow('Poker Card Edges', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
