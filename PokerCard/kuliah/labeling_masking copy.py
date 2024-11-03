import cv2
import numpy as np
import time
import os
import collections
import math
import matplotlib.pyplot as plt
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


def DrawCircle(image, y, x):
    center_coor = (x, y)
    radius = 2
    color = (255, 0, 0)
    thickness = 2
    image = cv2.circle(image, center_coor, radius, color, thickness)
    return image


def draw_labeled_box(image, x, y, w, h, label, color):
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    cv2.putText(image, f"Card {label}", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def four_point_transform(image, pts):
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect

    maxWidth = 200
    maxHeight = 300

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M, mask = cv2.findHomography(rect, dst)

    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def combine_images(images, cols=2):
    rows = (len(images) + cols - 1) // cols
    height = max(img.shape[0] for img in images)
    width = max(img.shape[1] for img in images)
    combined = np.zeros((height * rows, width * cols, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        combined[r*height:(r+1)*height, c*width:(c+1) *
                 width] = cv2.resize(img, (width, height))

    return combined


def extract_features(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (64, 64))

    features, _ = hog(image, orientations=8, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    return features


def get_corner_snip(flattened_images: list):
    corner_images = []
    for img in flattened_images:
        crop = img[5:110, 0:38]

        crop = cv2.resize(crop, None, fx=4, fy=4)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        bilateral = cv2.bilateralFilter(bin_img, 11, 174, 17)
        canny = cv2.Canny(bilateral, 40, 24)
        kernel = np.ones((1, 1))
        result = cv2.dilate(canny, kernel=kernel, iterations=2)

        corner_images.append([result, bin_img])

    return corner_images


def match_card(rank_image, suit_image, ranks_path, suits_path, threshold=0.8):
    best_rank_match = None
    best_rank_score = 0
    best_rank_name = ""

    best_suit_match = None
    best_suit_score = 0
    best_suit_name = ""

    for rank_file in os.listdir(ranks_path):
        rank_path = os.path.join(ranks_path, rank_file)
        template_image = cv2.imread(rank_path, 0)

        result = cv2.matchTemplate(
            rank_image, template_image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_rank_score and max_val >= threshold:
            best_rank_score = max_val
            best_rank_match = (rank_path, max_loc, template_image.shape)
            best_rank_name = rank_file

    for suit_file in os.listdir(suits_path):
        suit_path = os.path.join(suits_path, suit_file)
        template_image = cv2.imread(suit_path, 0)

        result = cv2.matchTemplate(
            suit_image, template_image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_suit_score and max_val >= threshold:
            best_suit_score = max_val
            best_suit_match = (suit_path, max_loc, template_image.shape)
            best_suit_name = suit_file

    result = {
        "rank": best_rank_name if best_rank_name else "no match",
        "suit": best_suit_name if best_suit_name else "no match"
    }

    return result


def preprocess_image(img, target_size):
    img = cv2.resize(img, target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi
    return img_array


def get_card_value(card):
    # Menentukan nilai berdasarkan nama kartu
    if "ace" in card:
        return 11  # Nilai Ace akan diatur kemudian jika melebihi 21
    elif any(face in card for face in ["jack", "queen", "king"]):
        return 10
    else:
        # Mengambil angka dari kartu (misalnya, "two", "three" diubah ke angka 2, 3, dll.)
        number_words = {
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        for word, value in number_words.items():
            if word in card:
                return value
    return 0  # Default jika tidak sesuai (misalnya, jika input salah)


def calculate_blackjack_score(cards):
    total_score = 0
    ace_count = 0

    # Hitung nilai setiap kartu
    for card in cards:
        value = get_card_value(card)
        total_score += value
        if value == 11:
            ace_count += 1

    # Penyesuaian untuk Ace jika total melebihi 21
    while total_score > 21 and ace_count:
        total_score -= 10  # Mengurangi nilai Ace dari 11 menjadi 1
        ace_count -= 1

    return total_score


model = load_model("kuliah/model.h5")  # Pastikan path model benar
dataset_path = "dataset/train"  # Pastikan path ke dataset benar
labels = sorted(os.listdir(dataset_path))
state = 0
prev_time = 0
count = 0
fps_history = collections.deque(maxlen=10)

# input_source = "play2_2.mp4"
input_source = "image.jpg"

cam = cv2.VideoCapture(input_source)


if not cam.isOpened() :
    print("Error opening camera")
    exit()

while True:
    if input_source.endswith(".jpg"):
        cam = cv2.VideoCapture(input_source)
    ret, frame = cam.read()
    if not ret:
        print("Error in retrieving frame")
        break

    flattened_cards = []
    cards = []

    if state == 0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([40, 40, 40])
        upper = np.array([80, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_not(mask)  # Membalik mask
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
        m = mask.copy()

        # Operasi erosi dan dilasi untuk mengurangi noise
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        foreground = cv2.bitwise_and(frame, frame, mask=mask)

        # Konversi foreground ke skala abu-abu untuk mencari kontur
        gray_foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)

        # Cari kontur pada foreground (citra biner)
        contours, _ = cv2.findContours(
            gray_foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fixed_width = 224
        fixed_height = 224

        for idx, contour in enumerate(contours):
            # Mengabaikan kontur kecil yang mungkin hanya noise
            if cv2.contourArea(contour) < 5000:
                continue

            # Dapatkan kotak rotasi minimal untuk setiap kontur
            rect = cv2.minAreaRect(contour)
            # Mendapatkan empat titik sudut dari kotak rotasi
            box = cv2.boxPoints(rect)
            box = np.int0(box)  # Mengubah koordinat ke tipe integer
            center = rect[0]
            center_x, center_y = center
            # to int
            center_x = int(center_x)
            center_y = int(center_y)

            # Gambar kotak di sekitar objek (kartu poker) sesuai rotasi

            # Mengurutkan titik sudut agar selalu menjadi urutan (top-left, top-right, bottom-right, bottom-left)
            # 1. Menjumlahkan koordinat untuk menentukan top-left dan bottom-right
            s = np.sum(box, axis=1)
            tl = box[np.argmin(s)]  # Top-left memiliki jumlah koordinat terkecil
            br = box[np.argmax(s)]  # Bottom-right memiliki jumlah koordinat terbesar

            # 2. Menghitung selisih koordinat untuk menentukan top-right dan bottom-left
            diff = np.diff(box, axis=1)
            tr = box[np.argmin(diff)]  # Top-right memiliki selisih terkecil
            bl = box[np.argmax(diff)]  # Bottom-left memiliki selisih terbesar

            # 3. Membuat array yang berisi titik sudut yang telah diurutkan
            ordered_pts = np.array([tl, tr, br, bl], dtype="float32")

            # Tentukan titik tujuan (corners) setelah kartu diluruskan dengan orientasi portrait
            dst_pts = np.array([[0, 0], [fixed_width - 1, 0], [fixed_width - 1,
                                                            fixed_height - 1], [0, fixed_height - 1]], dtype="float32")

            # Mendapatkan matriks transformasi perspektif
            M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)

            # Lakukan transformasi perspektif untuk meluruskan kartu
            flattened_card = cv2.warpPerspective(frame, M, (fixed_width, fixed_height))
            # Tampilkan hasil kartu yang telah diluruskan

            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

            # flattened card to rgb
            flattened_card = cv2.cvtColor(flattened_card, cv2.COLOR_BGR2RGB)

            # Ukuran flattened_card
            resized_card = cv2.resize(flattened_card, (150, 150))
            input_data = np.expand_dims(resized_card, axis=0)
            # plt.imshow(flattened_card)
            # plt.show()

            predictions = model.predict(input_data, verbose=0)
            class_id = np.argmax(predictions)
            confidence = predictions[0][class_id]
            label = labels[class_id]
            cards.append(label)

            text = f"{label}"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            text_x = center_x - (text_width // 2)  # Menempatkan teks di tengah sesuai lebar teks
            text_y = center_y - 10 
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        score = calculate_blackjack_score(cards)
        cv2.putText(frame, f"Score: {score}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    current_time = time.time()

    # Hitung FPS untuk frame ini
    fps = 1 / (current_time - prev_time)
    prev_time = current_time    
    count += 1
    if count == 1:
        avg_fps = fps
    else:
        avg_fps = math.ceil((avg_fps*count+fps)/(count+1))

    # Menampilkan rata-rata FPS pada frame
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(
        frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('1'): # jika 1 ditekan
        state = 0

cam.release()
cv2.destroyAllWindows()
