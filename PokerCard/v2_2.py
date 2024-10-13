import cv2
import pytesseract

# Path ke executable Tesseract (jika di Windows)
# Sesuaikan dengan lokasi instalasi tesseract di komputer Anda
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\agusf\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Step 1: Load the image
image = cv2.VideoCapture(0)

while True:
    ret, frame = image.read()
    if not ret:
        break
    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    

    # Step 3: Apply thresholding to make the text clearer for OCR
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Step 4: Detect text using Tesseract OCR
    # Lang parameter digunakan untuk memilih bahasa yang digunakan Tesseract (optional)
    # Anda dapat menambahkan 'eng' (Bahasa Inggris) untuk meningkatkan akurasi
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    # Draw bounding boxes around the detected text on the original copy of the image
    data = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)
    for i in range(len(data['level'])):
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Step 5: Print the detected text
    print("Detected Text: ", text)

    # Step 6: Display the image with the detected text region
    cv2.imshow('Processed Image for OCR', thresh)
    cv2.imshow('Original Image', frame)
        # Step 12: Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 13: Release the video capture object and close all OpenCV windows
image.release()
cv2.destroyAllWindows()

