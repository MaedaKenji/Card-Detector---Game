import cv2
import numpy as np


def DrawCircle(image, y, x):
    """
    Fungsi untuk menggambar lingkaran pada gambar pada koordinat tertentu.
    """
    center_coor = (x, y)  # Koordinat pusat lingkaran (x, y)
    radius = 2  # Jari-jari lingkaran
    color = (255, 0, 0)  # Warna lingkaran (Biru dalam format BGR)
    thickness = 2  # Ketebalan garis lingkaran

    # Menggambar lingkaran pada gambar
    image = cv2.circle(image, center_coor, radius, color, thickness)
    return image


def draw_labeled_box(image, x, y, w, h, label, color):
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    cv2.putText(image, f"Card {label}", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
def four_point_transform(image, pts):
    # Urutkan titik-titik (kiri atas, kanan atas, kanan bawah, kiri bawah)
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect

    # Hitung lebar maksimum
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Hitung tinggi maksimum
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Tentukan titik-titik tujuan untuk transformasi perspektif
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Hitung matriks transformasi perspektif
    M = cv2.getPerspectiveTransform(rect, dst)

    # Terapkan transformasi perspektif
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# Fungsi untuk menggabungkan beberapa gambar dalam satu grid
def combine_images(images, cols=2):
    rows = (len(images) + cols - 1) // cols
    height = max(img.shape[0] for img in images)
    width = max(img.shape[1] for img in images)
    combined = np.zeros((height * rows, width * cols, 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        combined[r*height:(r+1)*height, c*width:(c+1)*width] = cv2.resize(img, (width, height))
    
    return combined


# Variabels
colors = np.random.randint(0, 255, size=(10, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]  # Warna hitam untuk background

# Akses kamera
# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture("dump.mp4")
if not cam.isOpened():
    print("Error opening camera")
    exit()

while True:
    # Ambil frame dari kamera
    ret, frame = cam.read()
    if not ret:
        print("Error in retrieving frame")
        break

    flattened_cards = []
    
    # Konversi frame ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Batas bawah dan atas warna hijau dalam HSV
    lower = np.array([40, 40, 40])
    upper = np.array([80, 255, 255])

    # Buat mask untuk deteksi warna hijau
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(mask)  # Membalik mask

    # Kernel untuk operasi morfologi
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)

    # Salin mask
    m = mask.copy()

    # Operasi erosi dan dilasi untuk mengurangi noise
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Ambil hanya bagian foreground dari frame
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    # Komponen yang terhubung dalam mask
    num_labels, labels_im = cv2.connectedComponents(mask)

    # Loop melalui setiap komponen terhubung (kecuali background)
    for i in range(1, num_labels):
        b, k = np.where(labels_im == i)

        # Mendapatkan batas-batas dari komponen terhubung
        bmin = b.min()
        bmax = b.max()
        kmin = k.min()
        kmax = k.max()

        xbmin = k[np.where(b == bmin)[0][0]]
        ykmin = b[np.where(k == kmin)[0][0]]
        ykmax = b[np.where(k == kmax)[0][0]]

        bottom_right_indices = np.where((b == bmax) & (k == kmax))[0]
        if len(bottom_right_indices) > 0:
            xbmax = kmax
            ybmax = bmax
        else:
            # Jika titik (kmax, bmax) tidak ada dalam komponen, ambil titik terdekat
            distances = (k - kmax)**2 + (b - bmax)**2
            nearest_index = np.argmin(distances)
            xbmax = k[nearest_index]
            ybmax = b[nearest_index]

        # Gambar lingkaran di sekitar batas-batas komponen (y,x)
        frame = DrawCircle(frame, bmin, xbmin) # kiri atas
        frame = DrawCircle(frame, bmax, xbmax) # kanan bawah
        frame = DrawCircle(frame, ybmax, xbmin) # kiri bawah
        frame = DrawCircle(frame, ykmax, kmax)  # kanan atas
        
        # titik-titik sudut
        pts = np.array([
            [xbmin, bmin],   # kiri atas
            [kmax, ykmax],   # kanan atas
            [xbmax, bmax],   # kanan bawah
            [xbmin, ybmax]   # kiri bawah
        ], dtype="float32")
        # Terapkan transformasi perspektif
        warped = four_point_transform(frame, pts)
        # Tambahkan kartu yang telah diluruskan ke list
        flattened_cards.append(warped)

    

    # Labelling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    labeled_image = np.zeros(
        (frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    # Terapkan warna ke setiap komponen
    for i in range(1, num_labels):
        labeled_image[labels == i] = colors[i]

    card_count = 0
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Filter berdasarkan area untuk menghindari noise
        if area > 1000:  # Anda mungkin perlu menyesuaikan nilai ini
            card_count += 1
            draw_labeled_box(frame, x, y, w, h, card_count, colors[i].tolist())

    cv2.putText(frame, f"Cards: {card_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    # cv2.imshow('m', m)
    # cv2.imshow('fr', foreground)
    # cv2.imshow('labeled', labeled_image)
    if flattened_cards:
        combined_cards = combine_images(flattened_cards)
        cv2.imshow('Flattened Cards', combined_cards)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) == ord('q'):
        break

# Lepaskan resource kamera dan tutup semua jendela
cam.release()
cv2.destroyAllWindows()
