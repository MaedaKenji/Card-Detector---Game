import pygame
import cv2
import numpy as np

# Inisialisasi Pygame
pygame.init()

# Inisialisasi kamera
cap = cv2.VideoCapture("dump.mp4")

# Mendapatkan dimensi frame kamera
ret, frame = cap.read()
height, width = frame.shape[:2]

# Menghitung dimensi baru dengan height 640 dan mempertahankan aspect ratio
aspect_ratio = width / height


# Membuat layar Pygame
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Camera Feed")

# Membuat objek Clock untuk mengontrol frame rate
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Membaca frame dari kamera
    ret, frame = cap.read()
    
    if ret:
        # Konversi frame OpenCV ke format Pygame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        
        # Menampilkan frame pada layar Pygame
        screen.blit(frame, (0, 0))
        
        # Update layar
        pygame.display.flip()
    
    # Mengontrol frame rate
    clock.tick(30)

# Membersihkan resources
cap.release()
pygame.quit()