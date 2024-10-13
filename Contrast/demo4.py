


imageBG = cv2.imread('Contrast/itsSurabaya2.jpg') # citra latar belakang




hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([30-15, 50, 0]) # (derajat hsv,Saturasi,Intensitas)
upper = np.array([30+15, 255, 255]) # (derajat hsv,Saturasi,Intensitas)
mask = cv2.inRange(hsv,lower,upper)
maskBG = cv2.bitwise_not(mask)
foreground = cv2.bitwise_and(img, img, mask=mask)
foregroundNot = cv2.bitwise_not(img, img, mask=mask)
background = cv2.bitwise_and(imageBG, imageBG, mask=maskBG)
imageBaru = cv2.bitwise_or(foreground, background)