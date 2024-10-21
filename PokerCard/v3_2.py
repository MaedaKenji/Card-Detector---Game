import cv2
import matplotlib.pyplot as plt
import numpy as np
from processing import process
from utils.Loader import Loader

def resize_with_aspect_ratio(image, width=None, height=None):
    # Get the original image dimensions
    h, w = image.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = w / h

    if width is None:
        # Calculate height based on the specified width
        new_height = int(height / aspect_ratio)
        resized_image = cv2.resize(image, (height, new_height))
    else:
        # Calculate width based on the specified height
        new_width = int(width * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, width))

    return resized_image


# cap = cv2.VideoCapture("poker_black.mp4")
cap = cv2.VideoCapture("pokerG2.mp4")
# cap = cv2.VideoCapture(1)

# Desired width and height
width = 640
height = 480

while True:
    success, img = cap.read()
    
    # If the video frame was not successfully captured, break the loop
    if not success:
        break
    
    # Resize the frame to 640x480
    # img_resized = cv2.resize(img, (width, height))
    img_resized = resize_with_aspect_ratio(img, width=640)
    
    # Convert to RGB for further processing
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Copy for further operations
    imgResult = img_rgb.copy()
    imgResult2 = img_rgb.copy()
    
    # Apply thresholding and find corners
    thresh = process.get_thresh(imgResult)
    corners_list = process.find_corners_set(thresh, imgResult, draw=True)
    four_corners_set = corners_list
    
    # plt.imshow(thresh)
    # plt.grid()
    # plt.show()
    
    for i, corners in enumerate(corners_list):
        top_left = corners[0][0]
        bottom_left = corners[1][0]
        bottom_right = corners[2][0]
        top_right = corners[3][0]
        
        # print(f'top_left: {top_left}')
        # print(f'bottom_left: {bottom_left}')
        # print(f'bottom_right: {bottom_right}')
        # print(f'top_right: {top_right}\n')
        
    flatten_card_set = process.find_flatten_cards(imgResult2, four_corners_set)

    for img_output in flatten_card_set:
        print(img_output.shape)
        cv2.imshow("Flatten Cards", img_output)
        # plt.imshow(img_output)
        # plt.show()
        
    cropped_images = process.get_corner_snip(flatten_card_set)
    for i, pair in enumerate(cropped_images):
        for j, img in enumerate(pair):
            # cv2.imwrite(f'num{i*2+j}.jpg', img)
            # plt.subplot(1, len(pair), j+1)
            # plt.imshow(img, 'gray')
            cv2.imshow('num', img)
            continue
            
        # plt.show()
        
    
    ranksuit_list: list = list()

    # plt.figure(figsize=(12, 6))
    for i, (img, original) in enumerate(cropped_images):

        drawable = img.copy()
        d2 = original.copy()

        contours, _ = cv2.findContours(drawable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cnts_sort = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        cnts_sort = sorted(cnts_sort, key=lambda x: cv2.boundingRect(x)[1])
        
        # for cnt in cnts_sort:
            # print(f'contour sorts = {cv2.contourArea(cnt)}')

        cv2.drawContours(drawable, cnts_sort, -1, (0, 255, 0), 1)

        # cv2.imwrite(f'{i}.jpg', drawable)
        # plt.grid(True)
        # plt.subplot(1, len(cropped_images), i+1)
        # plt.imshow(img)

        ranksuit = list()

        for i, cnt in enumerate(cnts_sort):
            x, y, w, h = cv2.boundingRect(cnt)
            x2, y2 = x+w, y+h

            crop = d2[y:y2, x:x2]
            if(i == 0): # rank: 70, 125
                crop = cv2.resize(crop, (70, 125), 0, 0)
            else: # suit: 70, 100
                crop = cv2.resize(crop, (70, 100), 0, 0)
            # convert to bin image
            _, crop = cv2.threshold(crop, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            crop = cv2.bitwise_not(crop)

            ranksuit.append(crop)
            # cv2.rectangle(d2, (x, y), (x2, y2), (0, 255, 0), 2)
        
        ranksuit_list.append(ranksuit)
        # cv2.imshow('', d2)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    # plt.show()
    
    
    black_img = np.zeros((120, 70))
    # plt.figure(figsize=(12, 6))
    for i, ranksuit in enumerate(ranksuit_list):

        rank = black_img
        suit = black_img
        try:
            rank = ranksuit[0]
            suit = ranksuit[1]
        except:
            pass

        # plt.subplot(len(ranksuit_list), 2, i*2+1)

        # cv2.imwrite(f"{i}.jpg", rank_name)
        # plt.imshow(rank, 'gray')
        # plt.subplot(len(ranksuit_list), 2, i*2+2)
        # plt.imshow(suit, 'gray')
    # plt.show()
    
    # train_ranks = Loader.load_ranks('imgs/ranks')
    train_ranks = Loader.load_ranks('imgs/ranks')
    # PokerCard/imgs/ranks
    train_suits = Loader.load_suits('imgs/suits')

    # print(train_ranks[0].img.shape)
    # print(train_suits[0].img.shape)

    # for i, rank in enumerate(train_ranks):
    #     plt.subplot(1, len(train_ranks), i +1)
    #     plt.axis('off')
    #     plt.imshow(rank.img, 'gray')

    # plt.show()

    # for i, suit in enumerate(train_suits):
    #     plt.subplot(1, len(train_suits), i +1)
    #     plt.axis('off')
    #     plt.imshow(suit.img, 'gray')

    # plt.show()
    
    for it in ranksuit_list:
        try:
            rank = it[0]
            suit = it[1]
        except:
            continue
        rs = process.template_matching(rank, suit, train_ranks, train_suits)
        print(rs)
    
    
    
    # Show the resized frame
    cv2.imshow("Resized Image", img_resized)
    cv2.imshow("Thresholding", imgResult2)
    
    
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
