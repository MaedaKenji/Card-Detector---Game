import random
import pygame
import sys
import cv2
import numpy as np
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
font_text = pygame.font.Font(None, 20)
situation = -1


# Define colors
WHITE = (255, 255, 255)
TRANSPARANT = (255, 0, 0, 0)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
GREEN = (34, 139, 34)
RED = (255, 0, 0)


background_image = pygame.image.load(
    r'C:\Local Disk E\Code\Python\PCV\PokerCard\imgs\bg.jpg')
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))
table_image = pygame.image.load(
    r'C:\Local Disk E\Code\Python\PCV\PokerCard\imgs\table.png')
table_image = pygame.transform.scale(
    table_image, (WIDTH, HEIGHT)).convert_alpha()
dealer_button_image = pygame.image.load(
    r'C:\Local Disk E\Code\Python\PCV\PokerCard\imgs\dealer-badge.png')
dealer_button_image = pygame.transform.scale(
    dealer_button_image, (20, 20)).convert_alpha()

dataset_folder = r"C:\Local Disk E\Code\Python\PCV\PokerCard\dataset\all_kartu"

card_files = [f for f in os.listdir(
    dataset_folder) if f.endswith(".png") or f.endswith(".jpg")]
river_cards = card_files
print(card_files)

blue_back = "blue_back.png"
card_width = 150
card_height = 200


CARD_WIDTH, CARD_HEIGHT = 70, 100
BORDER_WIDTH = 2
CARD_COLOR = (255, 255, 255)  # Putih
BORDER_COLOR = (255, 215, 0)  # Emas

TIMER_DURATION = 10  # 10 detik


back_path = r"C:\Local Disk E\Code\Python\PCV\PokerCard\dataset\back"
back_image = pygame.image.load(os.path.join(back_path, "blue_back.png"))
back_images = pygame.transform.scale(
    back_image, (CARD_WIDTH-BORDER_WIDTH, CARD_HEIGHT-BORDER_WIDTH))

winner_image = pygame.image.load(
    r"C:\Local Disk E\Code\Python\PCV\PokerCard\imgs\win.jpg")
winner_image = pygame.transform.scale(winner_image, (WIDTH, HEIGHT))

lose_image = pygame.image.load(
    r"C:\Local Disk E\Code\Python\PCV\PokerCard\imgs\lose.jpg")
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
    "king of spades": "KS.png"
}


# Button dimensions
button_width, button_height = 200, 80

# Button positions
play_button_rect = pygame.Rect(
    (WIDTH // 2 - button_width // 2, HEIGHT //
     2 - 100), (button_width, button_height)
)
exit_button_rect = pygame.Rect(
    (WIDTH // 2 - button_width // 2, HEIGHT //
     2 + 50), (button_width, button_height)
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
text_rect = text_surface.get_rect(center=(80, 150))


def draw_timer_button(x, y, width, height, progress, total_duration):
    pygame.draw.rect(screen, BLACK, (x, y, width, height),
                     2)  # Garis luar tombol
    filled_width = (progress / total_duration) * width
    pygame.draw.rect(screen, RED, (x, y, filled_width, height)
                     ) 



def draw_card_with_border(surface, x, y):
    # Gambar border
    pygame.draw.rect(
        surface,
        BORDER_COLOR,
        (x, y, CARD_WIDTH, CARD_HEIGHT)
    )
    # Gambar kartu di dalam border
    pygame.draw.rect(
        surface,
        CARD_COLOR,
        (x + BORDER_WIDTH, y + BORDER_WIDTH, CARD_WIDTH -
         2 * BORDER_WIDTH, CARD_HEIGHT - 2 * BORDER_WIDTH)
    )


def draw_button(text, rect, color):
    pygame.draw.rect(screen, color, rect)
    label = font.render(text, True, WHITE)
    label_rect = label.get_rect(center=rect.center)
    screen.blit(label, label_rect)
    

def draw_button_font(text, rect, color,font):
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
                    print("Game Start")
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
    # input_source = 0
    # input_source = "C:\Local Disk E\Code\Python\PCV\PokerCard\dump.mp4"
    # input_source = r"C:\Local Disk E\Code\Python\PCV\PokerCard\imgs\20.jpg"

    pygame.time.get_ticks()  # Waktu mulai
    frame_height, frame_width = 0, 0
    cam = cv2.VideoCapture(input_source)
    model = load_model(
        "C:\Local Disk E\Code\Python\PCV\PokerCard\kuliah\model2.h5")
    dataset_path = r"C:\Local Disk E\Code\Python\PCV\PokerCard\dataset\train"
    labels = sorted(os.listdir(dataset_path))

    cards = []
    captured_cards = []
    player_cards = []
    bot1_cards = []
    bot2_cards = []
    bot3_cards = []
    bot4_cards = []
    dealer_coordinates = [(210, 155), (1090, 155),
                          (197, 575), (640, 575), (1090, 575)]
    timer_coordinates = [(210, 155), (1090, 155),
                         (197, 575), (590, 725), (1090, 575)]

    is_pygame = 0
    state = 1
    money_player = 10000
    money_bot1 = 10000
    money_bot2 = 10000
    money_bot3 = 10000
    money_bot4 = 10000

    pot = 0
    is_raise = 0
    is_raised = 0
    is_fold = 0
    is_call = 0
    global river_cards
    
    display_is_update = False

    turn_player = 0
    last_player_turn = turn_player
    phase_game = 0
    river_cards_images = None
    bot1_cards_images = None
    bot2_cards_images = None
    bot3_cards_images = None
    bot4_cards_images = None
    player_cards_images = None
    river_cards_selected = None
    
    input_text = ""
    input_font = pygame.font.Font(None, 22)
    ok_button_rect = pygame.Rect(693, 523, 47, 37)
    ok_active = 0
    active = False
    input_box = pygame.Rect(565, 523, 100, 37)

    small_blind = 50
    big_blind = 100
    current_bet = 0
    call_amount = 0

    blind_placed = False

    last_execution_time = 0
    last_update_time = time.time() 
    progress = 0  
    total_duration = 5  
    CALL_EXECUTION_INTERVAL = 2
    
    random.shuffle(card_files)  

    def load_card_image(card_file):
        card_image = pygame.image.load(os.path.join(dataset_folder, card_file))
        return pygame.transform.scale(card_image, (CARD_WIDTH-BORDER_WIDTH, CARD_HEIGHT-BORDER_WIDTH))

    # Fungsi untuk mengatur blind di awal permainan
    def set_initial_blinds():
        nonlocal money_player, money_bot1, money_bot2, money_bot3, money_bot4, pot, blind_placed, turn_player

        if not blind_placed:
            # Kurangi chip dari setiap pemain untuk small blind dan big blind
            money_player -= small_blind
            money_bot1 -= small_blind
            money_bot2 -= small_blind
            money_bot3 -= small_blind
            money_bot4 -= small_blind

            # Tambahkan ke pot
            pot += small_blind * 4 + big_blind

            # Set turn player dimulai dari pemain setelah big blind
            turn_player = last_player_turn  # Bot1 yang memasang big blind

            # Set flag blind_placed agar tidak diulang
            blind_placed = True
            
            

            # Fungsi tambahan untuk konversi

    def card_to_standard_format(card):
        """Konversi kartu dari berbagai format ke format standar (2C, 5D, dll)"""

        # Jika kartu mengandung ekstensi .png, hapus ekstensi tersebut
        if card.endswith('.png'):
            # Hapus bagian setelah '.' (termasuk ekstensi .png)
            card = card.split('.')[0]

        # Jika kartu sudah dalam format 2C, 5D, dll, kembalikan langsung
        if card[0] in '23456789TJQKA' and card[1] in 'CDHS':
            return card

        # Jika kartu dimulai dengan "10", ganti dengan "T" untuk format standar
        if card.startswith("10"):
            return "T" + card[2:]

        # Konversi dari format 'five of diamonds' ke '5D'
        rank_map = {
            'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': 'T', 'jack': 'J', 'queen': 'Q', 'king': 'K', 'ace': 'A'
        }
        suit_map = {
            'clubs': 'C', 'diamonds': 'D', 'hearts': 'H', 'spades': 'S'
        }

        # Pisahkan rank dan suit dari format 'five of diamonds' atau lainnya
        parts = card.split(' of ')
        # Pastikan menjadi huruf kecil agar tidak case-sensitive
        rank = rank_map[parts[0].lower()]
        suit = suit_map[parts[1].lower()]  # Sama dengan suit, pastikan huruf kecil
        return rank + suit

    
    def get_card_rank(card):
        """Mendapatkan peringkat kartu"""
        rank_order = '23456789TJQKA'

        # Jika kartu memiliki ekstensi file atau spasi, bersihkan kartu
        # card = card.strip()  # Hilangkan spasi di awal/akhir
        if '.' in card:
            card = card.split('.')[0]

        # Validasi panjang kartu
        # if len(card) != 2:
        #     raise ValueError(f"Kartu tidak valid: {card}")

        rank = card[0]

        # Validasi rank kartu
        if rank not in rank_order:
            # raise ValueError(f"Rank kartu tidak valid: {rank}")
            return False

        return rank_order.index(rank)

    def is_straight(cards):
        """Cek apakah kartu adalah straight"""
        # print(cards)

        # Jika kartu dalam format tuple (misalnya ('4D', '8C', ...), (2, 'One Pair')),
        # ambil hanya elemen pertama yang berisi daftar kartu.
        if isinstance(cards, tuple):
            cards = cards[0]

        ranks = []
        for card in cards:
            # if '.' in card:
            card = card.split('.')[0]  # Hapus ekstensi file jika ada
            rank = get_card_rank(card)  # Ambil peringkat kartu
            if rank is False:
                continue
            ranks.append(rank)

        ranks = list(set(ranks))  # Hapus duplikasi
        ranks.sort()

        # Periksa jika ada lima kartu berurutan
        if len(ranks) < 5:
            return False

        # Cek untuk kombinasi straight biasa
        for i in range(len(ranks) - 4):
            if ranks[i + 4] - ranks[i] == 4:
                return True

        # Cek untuk kombinasi wheel straight (A, 2, 3, 4, 5)
        if 12 in ranks and {0, 1, 2, 3, 4}.issubset(set(ranks)):
            return True

        return False

    
    def get_card_suit(card):
        """Mendapatkan suit kartu"""
        return card[1]

    def is_flush(cards):
        """Cek apakah kartu adalah flush"""
        suits = [get_card_suit(card) for card in cards]
        return len(set(suits)) == 1


    def is_royal_flush(cards):
        """Cek apakah kartu adalah royal flush"""
        if not is_flush(cards):
            return False
        ranks = [get_card_rank(card) for card in cards]
        return set(ranks) == {8, 9, 10, 11, 12}  # 10, J, Q, K, A

    def is_straight_flush(cards):
        """Cek apakah kartu adalah straight flush"""
        return is_flush(cards) and is_straight(cards)

    def count_pairs(cards):
        """Hitung pasangan kartu"""
        rank_counts = {}
        for card in cards:
            rank = card[0]
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        return sorted(rank_counts.values(), reverse=True)


    def evaluate_hand(cards):
        """Evaluasi kekuatan tangan poker"""
        if is_royal_flush(cards):
            return (10, "Royal Flush")  # Royal Flush
        elif is_straight_flush(cards):
            return (9, "Straight Flush")  # Straight Flush

        pair_counts = count_pairs(cards)

        if pair_counts[0] == 4:
            return (8, "Four of a Kind")  # Four of a Kind
        elif pair_counts[0] == 3 and pair_counts[1] == 2:
            return (7, "Full House")  # Full House
        elif is_flush(cards):
            return (6, "Flush")  # Flush
        elif is_straight(cards):
            return (5, "Straight")  # Straight
        elif pair_counts[0] == 3:
            return (4, "Three of a Kind")  # Three of a Kind
        elif pair_counts[0] == 2 and pair_counts[1] == 2:
            return (3, "Two Pair")  # Two Pair
        elif pair_counts[0] == 2:
            return (2, "One Pair")  # One Pair
        else:
            return (1, "High Card")  # High Card
        

    def find_best_5_card_hand(cards):
        """Temukan kombinasi 5 kartu terbaik dari semua kartu"""
        from itertools import combinations

        best_hand = None
        best_score = -1
        best_description = None

        for hand in combinations(cards, 5):
            hand_score, hand_description = evaluate_hand(hand)
            if hand_score > best_score:  # Bandingkan berdasarkan skor
                best_hand = hand
                best_score = hand_score
                best_description = hand_description  # Simpan deskripsi tangan terbaik

        return best_hand, (best_score, best_description)

    
    def showdown(player_cards, river_cards):
        """Proses showdown dengan semua kartu"""
        # Gabungkan kartu pemain dengan kartu river
        all_cards = player_cards + river_cards

        # Cari kombinasi 5 kartu terbaik
        best_hand, (best_score, best_description) = find_best_5_card_hand(all_cards)

        return best_hand, (best_score, best_description)
    
    def get_winner(hands):
        # Urutkan tangan berdasarkan peringkat (lebih tinggi lebih baik)
        sorted_hands = sorted(
            hands, key=lambda x: evaluate_hand(x[1])[0], reverse=True)

        # Tentukan peringkat tertinggi
        top_rank = evaluate_hand(sorted_hands[0][1])[0]

        # Cek jika pemain dengan peringkat tertinggi memiliki tangan yang sama
        tied_hands = [sorted_hands[0]]

        # Bandingkan pemain lain dengan peringkat tertinggi hanya jika mereka memiliki peringkat yang sama
        for hand in sorted_hands[1:]:
            if evaluate_hand(hand[1])[0] == top_rank:
                tied_hands.append(hand)
            else:
                break  # Jika sudah ada pemain dengan peringkat lebih rendah, berhenti

        # Jika ada lebih dari satu pemain dengan peringkat yang sama, cek kicker
        if len(tied_hands) > 1:
            return resolve_tie(tied_hands)

        # Jika hanya satu pemenang, kembalikan pemenang
        return tied_hands[0]

    def resolve_tie(tied_hands):
        # Urutkan tangan berdasarkan rank utama (tangan terbaik dulu)
        tied_hands.sort(key=lambda x: evaluate_hand(x[1])[0], reverse=True)
        
        # Jika peringkat tangan pertama berbeda, tidak ada tie
        if evaluate_hand(tied_hands[0][1])[0] != evaluate_hand(tied_hands[1][1])[0]:
            return tied_hands[0]

        # Jika peringkat tangan sama, bandingkan kicker
        for i in range(4, -1, -1):  # Cek kicker dari yang tertinggi ke yang terendah
            # Urutkan tangan berdasarkan kicker (kartu ke-i)
            tied_hands.sort(key=lambda x: get_card_rank(x[1][0][i]), reverse=True)
            # Jika ada pemenang berdasarkan kicker
            if get_card_rank(tied_hands[0][1][0][i]) > get_card_rank(tied_hands[1][1][0][i]):
                return tied_hands[0]
            elif get_card_rank(tied_hands[0][1][0][i]) < get_card_rank(tied_hands[1][1][0][i]):
                return tied_hands[1]

        # Jika semua kartu kicker sama, berarti seri total
        return None  # Seri total


    def create_winner_popup(screen, winner_name, winner_hand, pot_amount):
        # Inisialisasi font
        pygame.font.init()
        
        # Warna
        WHITE = (255, 255, 255)
        GREEN = (0, 128, 0)
        GOLD = (255, 215, 0)

        # Ukuran popup
        popup_width = 400
        popup_height = 250
        
        # Posisi popup di tengah layar
        x = (screen.get_width() - popup_width) // 2
        y = (screen.get_height() - popup_height) // 2

        # Buat surface untuk popup
        popup_surface = pygame.Surface((popup_width, popup_height))
        popup_surface.fill(GREEN)
        
        # Font
        title_font = pygame.font.Font(None, 36)
        content_font = pygame.font.Font(None, 28)

        # Judul
        title_text = title_font.render("WINNER!", True, GOLD)
        title_rect = title_text.get_rect(center=(popup_width//2, 50))
        popup_surface.blit(title_text, title_rect)

        # Nama pemenang
        name_text = content_font.render(f"Player: {winner_name}", True, WHITE)
        name_rect = name_text.get_rect(center=(popup_width//2, 100))
        popup_surface.blit(name_text, name_rect)

        # Tangan pemenang
        hand_text = content_font.render("Hand: " + ' '.join(winner_hand), True, WHITE)
        hand_rect = hand_text.get_rect(center=(popup_width//2, 150))
        popup_surface.blit(hand_text, hand_rect)

        # Pot yang dimenangkan
        pot_text = content_font.render(f"Pot: ${pot_amount}", True, WHITE)
        pot_rect = pot_text.get_rect(center=(popup_width//2, 200))
        popup_surface.blit(pot_text, pot_rect)

        # Animasi fade in
        for alpha in range(0, 255, 15):
            popup_surface.set_alpha(alpha)
            screen.blit(popup_surface, (x, y))
            pygame.display.flip()
            pygame.time.delay(30)

        # Tampilkan popup selama beberapa detik
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < 3000:  # Tampil selama 3 detik
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            pygame.display.update()

        # Animasi fade out
        for alpha in range(255, 0, -15):
            popup_surface.set_alpha(alpha)
            screen.blit(popup_surface, (x, y))
            pygame.display.flip()
            pygame.time.delay(30)
            
            
            
        
# GAMEEEEEEEEEEEEEE ----------------------------------------------------------
    if not cam.isOpened():
        print("Error opening camera")
        exit()

    while True:
        # Display the frame on the Pygame screen
        screen.fill(GREEN)
        mouse_x, mouse_y = pygame.mouse.get_pos()

        

        if state == 1:
            is_pygame = 0
            if type(input_source) == str and input_source.endswith(".jpg"):  # noqa: E721
                cam = cv2.VideoCapture(input_source)
            
            ret, frame = cam.read()
            if not ret:
                print("Error in retrieving frame")

            flattened_cards = []
            cards = []
            

            if frame_height == 0 or frame_width == 0:
                frame_height, frame_width = frame.shape[:2]

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array([40, 40, 40])
            upper = np.array([80, 255, 255])

            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_not(mask)
            kernel = np.array(
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)

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

                dst_pts = np.array([[0, 0], [fixed_width - 1, 0], [fixed_width - 1,
                                   fixed_height - 1], [0, fixed_height - 1]], dtype="float32")

                mask = cv2.getPerspectiveTransform(ordered_pts, dst_pts)

                flattened_card = cv2.warpPerspective(
                    frame, mask, (fixed_width, fixed_height))
                flattened_cards.append(flattened_card)
                flattened_card = cv2.cvtColor(
                    flattened_card, cv2.COLOR_BGR2RGB)

                resized_card = cv2.resize(flattened_card, (150, 150))
                input_data = np.expand_dims(resized_card, axis=0)

                predictions = model.predict(input_data, verbose=0)
                class_id = np.argmax(predictions)
                confidence = predictions[0][class_id]
                label = labels[class_id]
                if confidence >= 0.6:
                    cards.append(label)
                
                # Cek apakah cards sudah ada di captured_cards
                if not any(card in captured_cards for card in cards) and captured_cards != 'joker':
                    captured_cards.extend(cards)
                    print(captured_cards)

                if len(captured_cards) >= 10:
                    state = 0
                    for _ in range(2):  # Setiap pemain mendapatkan 2 kartu
                        # Pilih kartu secara acak untuk masing-masing pemain dan bot
                        player_cards.append(captured_cards.pop(
                            random.randint(0, len(captured_cards) - 1)))
                        bot1_cards.append(captured_cards.pop(
                            random.randint(0, len(captured_cards) - 1)))
                        bot2_cards.append(captured_cards.pop(
                            random.randint(0, len(captured_cards) - 1)))
                        bot3_cards.append(captured_cards.pop(
                            random.randint(0, len(captured_cards) - 1)))
                        bot4_cards.append(captured_cards.pop(
                            random.randint(0, len(captured_cards) - 1)))

                    player_cards_text = [card_to_image.get(
                        card) for card in player_cards]
                    bot1_cards_text = [card_to_image.get(
                        card) for card in bot1_cards]
                    bot2_cards_text = [card_to_image.get(
                        card) for card in bot2_cards]
                    bot3_cards_text = [card_to_image.get(
                        card) for card in bot3_cards]
                    bot4_cards_text = [card_to_image.get(
                        card) for card in bot4_cards]

                    # Kartu pemain
                    player_cards_images = [load_card_image(
                        card) for card in player_cards_text]
                    bot1_cards_images = [load_card_image(
                        card) for card in bot1_cards_text]
                    bot2_cards_images = [load_card_image(
                        card) for card in bot2_cards_text]
                    bot3_cards_images = [load_card_image(
                        card) for card in bot3_cards_text]
                    bot4_cards_images = [load_card_image(
                        card) for card in bot4_cards_text]

                    # RIVER
                    all_players_cards = player_cards_text + bot1_cards_text + \
                        bot2_cards_text + bot3_cards_text + bot4_cards_text
                    river_cards = [
                        card for card in river_cards if card not in all_players_cards]
                    river_cards_selected = random.sample(river_cards, 5)

                    river_cards_images = [load_card_image(
                        card) for card in river_cards_selected]

                    # Update state
                    is_pygame = 1
                    last_update_time = time.time()
                    # phase_game = 1
                    
                set_initial_blinds()

                    

                text = f"{label}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                text_x = center_x - (text_width // 2)
                text_y = center_y - 10
                cv2.putText(frame, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.image.frombuffer(
                frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
            screen.blit(frame_surface, (300, 100))


        if is_pygame == 1:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # Waktu berlalu dalam detik
            current_time = time.time()

            # BACK IMAGE
            draw_card_with_border(screen, 583, 52)
            screen.blit(back_images, (583, 52))

            # RIVER
            pot_text = f"Pot: {pot}"
            pot_text = font_text.render(pot_text, True, WHITE)
            screen.blit(pot_text, (584, 387))
            for i in range(5):
                draw_card_with_border(screen, 436+(i*75), 273)
                if phase_game == 1 and i < 3:
                    screen.blit(river_cards_images[i], (436+(i*75), 273))
                elif phase_game == 2 and i < 4:
                    screen.blit(river_cards_images[i], (436+(i*75), 273))
                elif phase_game == 3 and i < 5:
                    screen.blit(river_cards_images[i], (436+(i*75), 273))

            # BOT 1
            for i in range(2):
                draw_card_with_border(screen, 140+(i*75), 52)
                screen.blit(bot1_cards_images[i], (140+(i*75), 52))
            bot1_money_text = font_text.render(str(money_bot1), True, WHITE)
            screen.blit(bot1_money_text, (140, 32))

            # BOT 2
            for i in range(2):
                draw_card_with_border(screen, 1032+(i*75), 52)
                screen.blit(bot2_cards_images[i], (1032+(i*75), 52))
            bot2_money_text = font_text.render(str(money_bot2), True, WHITE)
            screen.blit(bot2_money_text, (1032, 32))

            # BOT 3
            for i in range(2):
                draw_card_with_border(screen, 140+(i*75), 602)
                screen.blit(bot3_cards_images[i], (140+(i*75), 602))
            bot3_money_text = font_text.render(str(money_bot3), True, WHITE)
            screen.blit(bot3_money_text, (140, 705))

            # BOT 4
            for i in range(2):
                draw_card_with_border(screen, 1032+(i*75), 602)
                screen.blit(bot4_cards_images[i], (1032+(i*75), 602))
            bot4_money_text = font_text.render(str(money_bot4), True, WHITE)
            screen.blit(bot4_money_text, (1032, 705))

            # PLAYER
            for i in range(2):
                draw_card_with_border(screen, 583+(i*75), 602)
                screen.blit(player_cards_images[i], (583+(i*75), 602))
            player_money_text = font_text.render(
                str(money_player), True, WHITE)
            screen.blit(player_money_text, (583, 705))

            # DEALER BUTTON
            if turn_player == 0:
                screen.blit(dealer_button_image, dealer_coordinates[3])
            elif turn_player == 1:
                screen.blit(dealer_button_image, dealer_coordinates[2])
            elif turn_player == 2:
                screen.blit(dealer_button_image, dealer_coordinates[0])
            elif turn_player == 3:
                screen.blit(dealer_button_image, dealer_coordinates[1])
            elif turn_player == 4:
                screen.blit(dealer_button_image, dealer_coordinates[4])
                

            if turn_player == 0 and is_fold == 0 and is_raise == 0:
                if not active:
                    progress = current_time - last_update_time
                else:
                    last_update_time += current_time - last_update_time 
                    
                
                if progress >= total_duration:
                    turn_player += 1  # Ubah ke pemain berikutnya
                    last_update_time = current_time  # Reset waktu terakhir
                    progress = 0  # Reset progres timer
                    is_fold = True
                
                draw_timer_button(timer_coordinates[3][0], timer_coordinates[3][1], 100, 20, progress, total_duration)
            elif turn_player != 0:
                last_update_time = current_time
                
            # BOT TIMER INTERVAL, bot interval
            
            woy = current_time - last_execution_time
            if turn_player != 0 and woy >= CALL_EXECUTION_INTERVAL:
                # turn_player += 1
                # Logika dasar
                hand_strength_1 = random.uniform(0, 1)
                hand_strength_2 = random.uniform(0, 1)
                hand_strength_3 = random.uniform(0, 1)
                hand_strength_4 = random.uniform(0, 1)
                
                if turn_player == 1:
                    if hand_strength_1 >= 0.3 and is_raised == 1:
                        money_bot1 -= call_amount
                elif turn_player == 2:
                    if hand_strength_2 >= 0.3 and is_raised == 1:
                        money_bot2 -= call_amount
                elif turn_player == 3:
                    if hand_strength_3 >= 0.3 and is_raised == 1:
                        money_bot3 -= call_amount
                elif turn_player == 4:
                    if hand_strength_4 >= 0.3 and is_raised == 1:
                        money_bot4 -= call_amount
                
                
                turn_player += 1
                last_execution_time = current_time
            elif turn_player == 0:
                last_execution_time = current_time
                
           
            
            
                
                
            
            
            
            if is_fold == 1:
                turn_player += 1
                
            if is_raise == 1:
                if ok_active == 1:
                    total_bet = pot + int(input_text)
                    
                    if total_bet > money_player:
                        print("NOT ENOUGH MONEY")    
                    else:
                        money_player -= int(input_text)
                        pot += int(input_text)
                        call_amount = int(input_text)
                        turn_player += 1
                        ok_active = 0
                        is_raise = 0
                    
                    
            # Input text
            if is_raise == 1:
                # Update Text
                text_surface = input_font.render(input_text, True, BLACK)
                screen.blit(text_surface, (input_box.x + 5, input_box.y + 5))
                
                # Gambar kotak input
                pygame.draw.rect(screen, GRAY if active else BLACK, input_box, 2)
                draw_button_font("Ok", ok_button_rect, GRAY, input_font)

                

            
            # BUTTON
            draw_button("Fold", fold_button_rect, GRAY)
            draw_button("Call", call_button_rect, GRAY)
            draw_button("Raise", raise_button_rect, GRAY)

            if pot == 0:
                set_initial_blinds()            

            if turn_player >= 5 and is_raise == 0:
                turn_player = 0
                phase_game += 1

            elif turn_player >= 5 and is_raise == 1:
                print("RAISED")

            if is_pygame == 1 and phase_game == 1 and not blind_placed:
                set_initial_blinds()

            if phase_game >= 4 and river_cards_selected is not None and display_is_update is True:
                # Konversi kartu dari format asli ke format 2C, dll
                player_standard_cards = [
                    card_to_standard_format(card) for card in player_cards]
                river_standard_cards = [card_to_standard_format(
                    card) for card in river_cards_selected]
                bot1_standard_cards = [
                    card_to_standard_format(card) for card in bot1_cards]
                bot2_standard_cards = [
                    card_to_standard_format(card) for card in bot2_cards]
                bot3_standard_cards = [
                    card_to_standard_format(card) for card in bot3_cards]
                bot4_standard_cards = [
                    card_to_standard_format(card) for card in bot4_cards]

                # Bandingkan tangan pemain
                player_hand = showdown(player_standard_cards, river_standard_cards)
                bot1_hand = showdown(bot1_standard_cards, river_standard_cards)
                bot2_hand = showdown(bot2_standard_cards, river_standard_cards)
                bot3_hand = showdown(bot3_standard_cards, river_standard_cards)
                bot4_hand = showdown(bot4_standard_cards, river_standard_cards)

                # Daftar tangan pemain dan bot
                hands = [
                    ('Player', player_hand),
                    ('Bot1', bot1_hand),
                    ('Bot2', bot2_hand),
                    ('Bot3', bot3_hand),
                    ('Bot4', bot4_hand)
                ]

                # Tentukan pemenang
                winner = get_winner(hands)

                # Jika ada seri, menangani hal tersebut
                if winner:
                    winner_name = winner[0]
                    winner_hand_rank = winner[1][1][1]
                else:
                    winner_name = "Tie"
                    winner_hand_rank = "No winner (Tie)"

                # Tampilkan pemenang atau seri menggunakan popup
                create_winner_popup(
                    screen,           # layar pygame
                    winner_name,      # nama pemenang atau "Tie"
                    # peringkat tangan pemenang atau "No winner (Tie)"
                    winner_hand_rank,
                    pot               # jumlah pot
                )

                if winner_name == "Player":
                    money_player += pot

                elif winner_name == "Bot1":
                    money_bot1 += pot

                elif winner_name == "Bot2":
                    money_bot2 += pot

                elif winner_name == "Bot3":
                    money_bot3 += pot

                elif winner_name == "Bot4":
                    money_bot4 += pot

                pot = 0
                phase_game = 0
                turn_player = last_player_turn
                is_pygame = 0
                is_raise = 0
                is_fold = 0
                state = 1
                display_is_update = False
                blind_placed = False
                set_initial_blinds()


        # UNIVERSALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
                   
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN and active:
                if event.key == pygame.K_RETURN:
                    input_text = ""  # Reset input setelah Enter
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]  # Hapus karakter terakhir
                else:
                    input_text += event.unicode  # Tambah karakter baru
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Press 'q' to quit video
                    return False
                elif event.key == pygame.K_r:  # Press 'r' to restart
                    return True
                elif event.key == pygame.K_p:
                    # captured_cards = [PokerCard/dataset/eight of clubs PokerCard/dataset/test/eight of diamonds PokerCard/dataset/test/eight of hearts PokerCard/dataset/test/eight of spades PokerCard/dataset/test/five of clubs PokerCard/dataset/test/five of diamonds PokerCard/dataset/test/five of hearts PokerCard/dataset/test/five of spades PokerCard/dataset/test/four of clubs PokerCard/dataset/test/four of diamonds
                    captured_cards = ["eight of clubs", "eight of diamonds", "eight of hearts", "eight of spades",
                                      "five of clubs", "five of diamonds", "five of hearts", "five of spades", "four of clubs", "four of diamonds"]
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    active = not active
                else:
                    active = False
                
                if text_rect.collidepoint(event.pos):
                    print("Stand")
                    is_stand = 1
                    is_money = 0
                    break
                elif fold_button_rect.collidepoint(event.pos):
                    print("Fold")
                    turn_player += 1
                    is_fold = 1
                    is_money = 0
                    break
                elif call_button_rect.collidepoint(event.pos) and turn_player == 0:
                    print("Call")
                    is_call = 1
                    is_money = 0
                    turn_player += 1
                    break
                elif raise_button_rect.collidepoint(event.pos):
                    print("Raise")
                    is_raise = 1
                    is_money = 0
                    break
                if ok_button_rect.collidepoint(event.pos):
                    ok_active = 1
                else:
                    ok_active = 0

 
        pygame.display.flip()
        display_is_update = True


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
            print("Direset")
        else:
            break


if __name__ == "__main__":
    main()
