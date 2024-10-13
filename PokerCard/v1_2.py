import cv2
import numpy as np

# Step 1: Capture video from file or webcam
video = cv2.VideoCapture(0)  # You can also use 0 for webcam

while True:
    # Step 2: Read each frame from the video
    ret, frame = video.read()
    if not ret:
        break  # Break the loop if there are no frames left

    # Step 3: Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 4: Apply histogram equalization to improve contrast
    gray = cv2.equalizeHist(gray)

    # Step 5: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 6: Apply Otsu's thresholding for better edge detection
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 7: Apply morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Step 8: Find contours in the frame
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_cards = 0

    # Step 9: Loop over contours and filter by area and shape (aspect ratio)
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter contours by size (adjust thresholds according to your card size)
        if 1000 < area < 15000:  # Set upper limit to avoid detecting large areas like the entire frame
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # If the approximated contour has four points, it's likely a card
            if len(approx) == 4:
                # Get the bounding box to compute aspect ratio
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)

                # Check if the aspect ratio is close to that of a poker card
                if 0.6 < aspect_ratio < 0.75:  # More flexible range for poker card aspect ratio
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)  # Draw green contour around the card
                    detected_cards += 1  # Keep track of the number of detected cards

    # Step 10: Display the number of detected cards on the frame
    cv2.putText(frame, f'Detected Cards: {detected_cards}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Step 11: Display the frame with the detected poker cards
    cv2.imshow('Poker Card Detection', frame)

    # Step 12: Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 13: Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
