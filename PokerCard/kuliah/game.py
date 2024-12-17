import collections
import random
from turtle import delay
import pygame
import sys
import cv2
import numpy as np
import math
import time
import os
from tensorflow.keras.models import load_model


# Initialize pygame
pygame.init()

# Set up screen
WIDTH, HEIGHT = 1280, 760
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Main Menu")
font = pygame.font.Font(None, 50)
situation = -1



# Define colors
WHITE = (255, 255, 255)
TRANSPARANT = (255, 0, 0, 0)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)

background_image = pygame.image.load(r'C:\Local Disk E\Code\Python\PCV\PokerCard\imgs\bg.jpg')
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))
table_image = pygame.image.load(r'C:\Local Disk E\Code\Python\PCV\PokerCard\imgs\table.jpg')
table_image = pygame.transform.scale(table_image, (WIDTH, HEIGHT)).convert_alpha()
dataset_folder = r"C:\Local Disk E\Code\Python\PCV\PokerCard\dataset\all_kartu"
card_files = [f for f in os.listdir(dataset_folder) if f.endswith(".png") or f.endswith(".jpg")]
blue_back = "blue_back.png"
card_width = 150
card_height = 200




back_path = r"C:\Local Disk E\Code\Python\PCV\PokerCard\dataset\back"
back_image = pygame.image.load(os.path.join(back_path,"blue_back.png"))
back_images = pygame.transform.scale(back_image, (card_width, card_height))

winner_image = pygame.image.load(r"C:\Local Disk E\Code\Python\PCV\PokerCard\imgs\win.jpg")
winner_image = pygame.transform.scale(winner_image, (WIDTH, HEIGHT))

lose_image = pygame.image.load(r"C:\Local Disk E\Code\Python\PCV\PokerCard\imgs\lose.jpg")
lose_image = pygame.transform.scale(lose_image, (WIDTH, HEIGHT))




card_mapping = {
    '10C.png': 'ten of clubs',
    '10D.png': 'ten of diamonds',
    '10H.png': 'ten of hearts',
    '10S.png': 'ten of spades',
    '2C.png': 'two of clubs',
    '2D.png': 'two of diamonds',
    '2H.png': 'two of hearts',
    '2S.png': 'two of spades',
    '3C.png': 'three of clubs',
    '3D.png': 'three of diamonds',
    '3H.png': 'three of hearts',
    '3S.png': 'three of spades',
    '4C.png': 'four of clubs',
    '4D.png': 'four of diamonds',
    '4H.png': 'four of hearts',
    '4S.png': 'four of spades',
    '5C.png': 'five of clubs',
    '5D.png': 'five of diamonds',
    '5H.png': 'five of hearts',
    '5S.png': 'five of spades',
    '6C.png': 'six of clubs',
    '6D.png': 'six of diamonds',
    '6H.png': 'six of hearts',
    '6S.png': 'six of spades',
    '7C.png': 'seven of clubs',
    '7D.png': 'seven of diamonds',
    '7H.png': 'seven of hearts',
    '7S.png': 'seven of spades',
    '8C.png': 'eight of clubs',
    '8D.png': 'eight of diamonds',
    '8H.png': 'eight of hearts',
    '8S.png': 'eight of spades',
    '9C.png': 'nine of clubs',
    '9D.png': 'nine of diamonds',
    '9H.png': 'nine of hearts',
    '9S.png': 'nine of spades',
    'AC.png': 'ace of clubs',
    'AD.png': 'ace of diamonds',
    'AH.png': 'ace of hearts',
    'AS.png': 'ace of spades',
    'JC.png': 'jack of clubs',
    'JD.png': 'jack of diamonds',
    'JH.png': 'jack of hearts',
    'JS.png': 'jack of spades',
    'KC.png': 'king of clubs',
    'KD.png': 'king of diamonds',
    'KH.png': 'king of hearts',
    'KS.png': 'king of spades',
    'QC.png': 'queen of clubs',
    'QD.png': 'queen of diamonds',
    'QH.png': 'queen of hearts',
    'QS.png': 'queen of spades'
}
card_to_image = {
    "ace of clubs": "AC.png",
    "ace of diamonds": "AD.png",
    "ace of hearts": "AH.png",
    "ace of spades": "AS.png",
    "two of clubs": "2C.png",
    "two of diamonds": "2D.png",
    "two of hearts": "2H.png",
    "two of spades": "2S.png",
    "three of clubs": "3C.png",
    "three of diamonds": "3D.png",
    "three of hearts": "3H.png",
    "three of spades": "3S.png",
    "four of clubs": "4C.png",
    "four of diamonds": "4D.png",
    "four of hearts": "4H.png",
    "four of spades": "4S.png",
    "five of clubs": "5C.png",
    "five of diamonds": "5D.png",
    "five of hearts": "5H.png",
    "five of spades": "5S.png",
    "six of clubs": "6C.png",
    "six of diamonds": "6D.png",
    "six of hearts": "6H.png",
    "six of spades": "6S.png",
    "seven of clubs": "7C.png",
    "seven of diamonds": "7D.png",
    "seven of hearts": "7H.png",
    "seven of spades": "7S.png",
    "eight of clubs": "8C.png",
    "eight of diamonds": "8D.png",
    "eight of hearts": "8H.png",
    "eight of spades": "8S.png",
    "nine of clubs": "9C.png",
    "nine of diamonds": "9D.png",
    "nine of hearts": "9H.png",
    "nine of spades": "9S.png",
    "ten of clubs": "10C.png",
    "ten of diamonds": "10D.png",
    "ten of hearts": "10H.png",
    "ten of spades": "10S.png",
    "jack of clubs": "JC.png",
    "jack of diamonds": "JD.png",
    "jack of hearts": "JH.png",
    "jack of spades": "JS.png",
    "queen of clubs": "QC.png",
    "queen of diamonds": "QD.png",
    "queen of hearts": "QH.png",
    "queen of spades": "QS.png",
    "king of clubs": "KC.png",
    "king of diamonds": "KD.png",
    "king of hearts": "KH.png",
    "king of spades": "KS.png",
    "joker": "joker.png"  # jika ada gambar untuk joker
}


# Button dimensions
button_width, button_height = 200, 80

# Button positions
play_button_rect = pygame.Rect(
    (WIDTH // 2 - button_width // 2, HEIGHT // 2 - 100), (button_width, button_height)
)
exit_button_rect = pygame.Rect(
    (WIDTH // 2 - button_width // 2, HEIGHT // 2 + 50), (button_width, button_height)
)
stand_button_rect = pygame.Rect(
    (10, 150), (button_width-80, button_height)
)   

call_button_rect = pygame.Rect(
    (10, 250), (button_width-80, button_height)
)

raise_button_rect = pygame.Rect(
    (10, 350), (button_width-80, button_height)
)

fold_button_rect = pygame.Rect(
    (10, 450), (button_width-80, button_height)
)

stand_text = "STAND"
text_surface = font.render(stand_text, True, WHITE)
text_rect = text_surface.get_rect(center=(80, 150))  # Posisi tombol di tengah layar



def draw_button(text, rect, color):
    pygame.draw.rect(screen, color, rect)
    label = font.render(text, True, WHITE)
    label_rect = label.get_rect(center=rect.center)
    screen.blit(label, label_rect)

def main_menu():
    situation = -1
    while True:
        screen.fill(BLACK)
        screen.blit(background_image, (0, 0))
        

        # Draw buttons
        draw_button("Play", play_button_rect, GRAY)
        draw_button("Exit", exit_button_rect, GRAY)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if play_button_rect.collidepoint(event.pos):
                    screen.fill(BLACK)
                    print ("Game Start")
                    situation = 1
                    screen.fill(BLACK)
                    break
                elif exit_button_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()
        

        if situation == 1:
            break
        pygame.display.flip()
    screen.fill(BLACK)
    if situation == 1:
        play_game()

def play_game():
    input_source = "image.jpg"
    input_source = 0
    input_source = "C:\Local Disk E\Code\Python\PCV\PokerCard\play2_2.mp4"
    input_source = "C:\Local Disk E\Code\Python\PCV\PokerCard\dump.mp4"
    input_source = r"C:\Local Disk E\Code\Python\PCV\PokerCard\imgs\20.jpg"
    
    
    prev_time = 0
    count = 0
    avg_fps = 0  # FPS rata-rata
    is_pygame = 1
    is_money = 0
    fps_history = collections.deque(maxlen=10)
    start_time = time.time()
    windows_opened = {}
    is_stand = 0
    money = 1000
    
    
    frame_height, frame_width = 0,0
    
    
    cam = cv2.VideoCapture(input_source)
    
    model = load_model("C:\Local Disk E\Code\Python\PCV\PokerCard\kuliah\model2.h5")
    dataset_path = r"C:\Local Disk E\Code\Python\PCV\PokerCard\dataset\train"  
    labels = sorted(os.listdir(dataset_path))
    
    random.shuffle(card_files)  # Shuffling kartu untuk randomisasi
    selected_cards = [card_files[0], card_files[1]]
    def load_card_image(card_file):
        card_image = pygame.image.load(os.path.join(dataset_folder, card_file))
        return pygame.transform.scale(card_image, (card_width, card_height))
    card_images = [load_card_image(card) for card in selected_cards]
    dealer_mapping = [card_mapping.get(card) for card in selected_cards]
    
    
    if not cam.isOpened() :
        print("Error opening camera")
        exit()
    
    while True:
        # Display the frame on the Pygame screen
        screen.fill(WHITE)
        
        if type(input_source) == str and input_source.endswith(".jpg"):
                cam = cv2.VideoCapture(input_source)
        ret, frame = cam.read()
        if ret:
            state = 1
        if not ret:
            print("Error in retrieving frame")
            # cam.release()

        if state == 1:
            flattened_cards = []
            cards = []
            dealer_cards = []
            player_cards = []     
            current_time = time.time()
                        
            if frame_height == 0 or frame_width == 0:
                frame_height, frame_width = frame.shape[:2]
                
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array([40, 40, 40])
            upper = np.array([80, 255, 255])

            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_not(mask)
            kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)

            mask = cv2.erode(mask, kernel, iterations=2)
            mask = cv2.dilate(mask, kernel, iterations=2)

            foreground = cv2.bitwise_and(frame, frame, mask=mask)

            gray_foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)

            contours, _ = cv2.findContours(
                gray_foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            fixed_width = 224
            fixed_height = 224

            for idx, contour in enumerate(contours):
                if cv2.contourArea(contour) < 16000:
                    continue
                if cv2.contourArea(contour) > 70000:
                    continue

                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                center = rect[0]
                center_x, center_y = int(center[0]), int(center[1])
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                box = np.array(box)

                box = box[np.argsort(box[:, 1])]
                tl, tr = sorted(box[:2], key=lambda x: x[0])
                bl, br = sorted(box[2:], key=lambda x: x[0])

                if len({tuple(tl), tuple(tr), tuple(bl), tuple(br)}) != 4:
                    print("Corners are not unique, skipping this box.")

                ordered_pts = np.array([tl, tr, br, bl], dtype="float32")

                dst_pts = np.array([[0, 0], [fixed_width - 1, 0], [fixed_width - 1, fixed_height - 1], [0, fixed_height - 1]], dtype="float32")

                mask = cv2.getPerspectiveTransform(ordered_pts, dst_pts)

                flattened_card = cv2.warpPerspective(frame, mask, (fixed_width, fixed_height))
                flattened_cards.append(flattened_card)
                flattened_card = cv2.cvtColor(flattened_card, cv2.COLOR_BGR2RGB)

                resized_card = cv2.resize(flattened_card, (150, 150))
                input_data = np.expand_dims(resized_card, axis=0)

                predictions = model.predict(input_data, verbose=0)
                class_id = np.argmax(predictions)
                # confidence = predictions[0][class_id]
                label = labels[class_id]
                cards.append(label)
                

                text = f"{label}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                text_x = center_x - (text_width // 2)
                text_y = center_y - 10
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)


            # Periksa perubahan jumlah flattened_cards dan update jendela
            for i in range(len(flattened_cards)):
                # Jika jendela untuk flattened card belum ada, buat jendela baru
                if i not in windows_opened:
                    cv2.imshow(f"Flattened Card {i}", flattened_cards[i])
                    windows_opened[i] = True  # Tandai jendela sebagai terbuka

                else:
                    # Update jendela yang sudah ada
                    cv2.imshow(f"Flattened Card {i}", flattened_cards[i])

            # Menutup jendela untuk kartu yang sudah tidak ada
            for i in list(windows_opened.keys()):
                if i >= len(flattened_cards):
                    # Menutup jendela jika kartu tidak ada lagi di flattened_cards
                    cv2.destroyWindow(f"Flattened Card {i}")
                    del windows_opened[i]  # Hapus dari dictionary karena jendela sudah ditutup
                
            # Hitung FPS untuk frame ini
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            count += 1
            
            # Perbarui FPS rata-rata setiap 3 detik
            if current_time - start_time >= 1:
                avg_fps = fps
                start_time = current_time  # Reset waktu mulai untuk 3 detik berikutnya
                count = 0  # Reset hitungan frame
            
            # Menampilkan rata-rata FPS pada frame
            fps_text = f"FPS: {avg_fps:.2f}"
            cv2.putText(
                frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            if frame is not None:
                try:
                    cv2.imshow("Frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        cam.release()                
                    
                except Exception as e:
                    print(e)
            else :
                print("Error in retrieving frame")
        
        if is_pygame == 1:            
            mouse_x, mouse_y = pygame.mouse.get_pos()

            # Mencetak koordinat mouse
            print(f'Koordinat Mouse: ({mouse_x}, {mouse_y})')
           
        # Event handling to quit the video display
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Press 'q' to quit video
                    return False
                elif event.key == pygame.K_r:  # Press 'r' to restart 
                    return True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if text_rect.collidepoint(event.pos):
                    print("Stand")
                    is_stand = 1
                    is_money = 0
                    break
        # pygame.draw.rect(screen, GRAY, stand_button_rect, 2)
        screen.blit(table_image, (0, 0))
        screen.blit(text_surface, text_rect)
        screen.blit(card_images[0], (450, 100))  
        player_cards_text = [card_to_image.get(card) for card in cards]
        player_cards_image = [load_card_image(card) for card in player_cards_text]
        
        
        # Text Score Player
        player_score = calculate_blackjack_score(cards)
        
        score_text = font.render(f"Player Score: {player_score}",True, (255,255,255))
        screen.blit(score_text,(35,600))
        for i, card_image in enumerate(player_cards_image):
            screen.blit(card_image, (i * card_width+350, 500))
            
        money_text = font.render(f"Money: {money}",True, (255,255,255))
        screen.blit(money_text,(35,700))
        
        if is_stand:
            screen.blit(card_images[1],(630,100))
            dealer_score = calculate_blackjack_score(dealer_mapping)
            print(f"Dealer Score: {dealer_score}")
            score_text = font.render(f"Dealer Score: {dealer_score}",True, (255,255,255))
            screen.blit(score_text,(80,190))
            pygame.display.flip()
            time.sleep(3)
            is_stand = False
        else:
            screen.blit(back_images,(630,100))
            
        try:
            if dealer_score > 21 and player_score <= 21 and is_money == 0:
                # screen.blit(winner_image,(0,0))
                is_money = 1
                money += 100
                # cv2.destroyAllWindows()
                # cam.release()
            elif dealer_score < player_score and player_score <= 21 and is_money == 0:
                # screen.blit(winner_image,(0,0))
                is_money = 1
                money += 100
                # cv2.destroyAllWindows()
                # cam.release()
            # else:
            #     # screen.blit(lose_image,(0,0))
            #     state = 0
            #     is_money = 1
            #     money -= 100
                # cv2.destroyAllWindows()
                # cam.release()
            # cam.release()
        except:
            pass

        pygame.display.flip()

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


def stand():
    pass

def main():
    main_menu()
    while True:
        # if play_game():
        if play_game():
            print ("Direset")
        else:
            break

if __name__ == "__main__":
    main()
