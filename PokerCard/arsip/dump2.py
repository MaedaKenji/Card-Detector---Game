import cv2

# Buka file video
cap = cv2.VideoCapture('play1.mp4')

# Dapatkan frame rate asli video
fps = cap.get(cv2.CAP_PROP_FPS)

# Hitung interval waktu untuk kecepatan 1.5x
time_interval = 1000 / (fps * 1.5)  # dalam milidetik

# Inisialisasi waktu
current_time = 0

while True:
    # Set posisi frame berdasarkan waktu saat ini
    cap.set(cv2.CAP_PROP_POS_MSEC, current_time)
    
    ret, frame = cap.read()
    if not ret:
        break

    # Tampilkan frame
    cv2.imshow('Video 1.5x Speed', frame)

    # Tunggu 1 ms atau sampai tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update waktu untuk frame berikutnya
    current_time += time_interval

# Bersihkan
cap.release()
cv2.destroyAllWindows()