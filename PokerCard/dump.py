import pygame
import sys
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from processing import process
from utils.Loader import Loader
from mss import mss
from PIL import Image

# Inisialisasi Pygame
pygame.init()

# Pengaturan layar dan warna
WIDTH, HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Poker Game")
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
YELLOW = (255, 223, 0)

# Font
FONT = pygame.font.SysFont("Arial", 36)
SMALL_FONT = pygame.font.SysFont("Arial", 24)

# Suara
pass_sound = pygame.mixer.Sound("pass_sound.wav")
player_turn_sound = pygame.mixer.Sound("player_turn.wav")
woosh_sound = pygame.mixer.Sound("woosh.wav")

# Rank dan Suit
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["Spades", "Hearts", "Clubs", "Diamonds"]

# Variabel game
player_turn = None  # None, "player", "ai"
player_cards = []
ai_cards = []
last_card_played = None  # (rank, suit, player)
is_passed = False  # Menyimpan status pass
selesai_register = False

# Fungsi untuk membuat deck kartu
def create_deck():
    return [(rank, suit) for rank in RANKS for suit in SUITS]

# Fungsi untuk menggambar tombol


def draw_button(text, x, y, width, height, color):
    pygame.draw.rect(SCREEN, color, (x, y, width, height))
    button_text = FONT.render(text, True, BLACK)
    text_rect = button_text.get_rect(center=(x + width // 2, y + height // 2))
    SCREEN.blit(button_text, text_rect)

# Fungsi untuk menggambar kartu


def draw_cards(cards, x, y):
    for i, card in enumerate(cards):
        rank, suit = card
        card_text = SMALL_FONT.render(f"{rank} of {suit}", True, WHITE)
        SCREEN.blit(card_text, (x, y + i * 30))

# Fungsi untuk mengocok dan membagikan kartu


def deal_cards():
    global player_cards, ai_cards
    deck = create_deck()
    random.shuffle(deck)
    # player_cards = deck[:10]
    ai_cards = deck[10:20]

# Fungsi untuk menentukan giliran pertama
def suitPlayer():
    return random.choice(["player", "ai"])

# Fungsi untuk login page
def login_page():
    SCREEN.fill(GREEN)
    title_text = FONT.render("Welcome to Poker Game", True, WHITE)
    SCREEN.blit(title_text, (WIDTH // 2 -
                title_text.get_width() // 2, HEIGHT // 4))
    draw_button("Play", WIDTH // 2 - 75, HEIGHT // 2, 150, 50, YELLOW)

# Fungsi untuk halaman game
def game_page():
    SCREEN.fill(GREEN)
    game_text = FONT.render("Poker Game - Main Page", True, WHITE)
    SCREEN.blit(game_text, (WIDTH // 2 - game_text.get_width() // 2, 50))

    # Tampilkan kartu pemain
    draw_cards(player_cards, 50, HEIGHT - 300)

    # Tampilkan kartu terakhir yang dimainkan dan siapa yang memainkannya
    if last_card_played:
        rank, suit, player = last_card_played
        last_card_text = SMALL_FONT.render(
            f"Last Played: {rank} of {suit} by {player}", True, WHITE
        )
        SCREEN.blit(last_card_text, (WIDTH // 2 -
                    last_card_text.get_width() // 2, HEIGHT // 2 - 50))

    # Tampilkan jumlah kartu AI yang tersisa
    ai_cards_text = FONT.render(f"AI Cards Left: {len(ai_cards)}", True, WHITE)
    SCREEN.blit(ai_cards_text, (WIDTH - ai_cards_text.get_width() - 20, 20))

    # Tombol pass
    draw_button("Pass", WIDTH // 2 - 75, HEIGHT - 200, 150, 50, YELLOW)

    # Tampilkan giliran
    turn_text = FONT.render(f"Turn: {player_turn}", True, WHITE)
    SCREEN.blit(turn_text, (WIDTH // 2 - turn_text.get_width() // 2, 100))

# Fungsi untuk memainkan kartu
def play_card(card):
    global last_card_played, player_turn, is_passed
    if card[0] is not "Unknown" and card[1] is not "Unknown" and  (is_passed or compare_cards(card, last_card_played)) and card != last_card_played[0:2]:
        if is_passed: print("Pass")
        elif compare_cards(card, last_card_played): print("Compare")
        print(f"last_card_played: {last_card_played}")
        last_card_played = (*card, "player")
        print(f"mau remove: {card}")
        if card in player_cards: 
            player_cards.remove(card)
            player_turn = "ai"
            is_passed = False
            woosh_sound.play()
            print(f"Player played: {card[0]} of {card[1]}")
            return True
        else :
            print(f"Invalid move card not in player's hand: {card[0]} of {card[1]}")
            return False
    else:
        print(f"Invalid move: {card[0]} of {card[1]}")
        return False


# Fungsi untuk membandingkan kartu
def compare_cards(card1, card2):
    # print(f"card1: {card1}, card2: {card2}")
    if card2 is None:
        return True
    rank1, _ = card1
    rank2, _ = card2[:2]
    return RANKS.index(rank1) > RANKS.index(rank2)

# Fungsi AI bermain
def ai_play():
    global last_card_played, player_turn, is_passed
    playable_cards = [
        card for card in ai_cards if compare_cards(card, last_card_played)]
    if is_passed or playable_cards:
        chosen_card = random.choice(
            playable_cards) if playable_cards else ai_cards[0]
        last_card_played = (*chosen_card, "AI")
        ai_cards.remove(chosen_card)
        player_turn = "player"
        player_turn_sound.play()
    else:
        is_passed = True
        player_turn = "player"
        player_turn_sound.play()

# Resize image with aspect ratio
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

def register_card(rs):
    global player_cards, selesai_register
    # print(f"Selesai register status: {selesai_register}")
    
    if rs and len(player_cards) < 10:
        rank, suit = rs
        
        # Periksa apakah rank atau suit adalah 'Unknown'
        if rank == 'Unknown' or suit == 'Unknown':
            print(f"Skipping unknown card: {rank} of {suit}")
            return len(player_cards)
        
        new_card = (rank, suit)
        
        # Periksa apakah kartu sudah ada di player_cards
        if new_card not in player_cards:
            player_cards.append(new_card)
            print(f"Added {rank} of {suit} to player's hand")
            for card in player_cards:
                print(card)
        else:
            print(f"{rank} of {suit} is already in player's hand")
    
    # Jika player_cards sudah memiliki 10 kartu
    if len(player_cards) >= 3:
        selesai_register = True
        print("Player's hand is full (10 cards)")
    
    return len(player_cards)  # Mengembalikan jumlah kartu di tangan pemain



# Fungsi utama
def main():
    global player_turn

    clock = pygame.time.Clock()
    player_turn = suitPlayer()
    deal_cards()

    # CV2
    cap = cv2.VideoCapture("play1.mp4")
    # cap = cv2.VideoCapture(1)
    
    width = 640
    height = 480

    while True:
        # CV2
        success, img = cap.read()
        rs = 0
        if not success:
            print("Error reading video file")
            
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
        flatten_card_set = process.find_flatten_cards(imgResult2, four_corners_set)

        if flatten_card_set is not None:
            for img_output in flatten_card_set:
        #         # print(img_output.shape)
                cv2.imshow("Flatten Cards", img_output)
            
            
        cropped_images = process.get_corner_snip(flatten_card_set)
        for i, pair in enumerate(cropped_images):
            for j, img in enumerate(pair):
                # cv2.imwrite(f'num{i*2+j}.jpg', img)
                # plt.subplot(1, len(pair), j+1)
                # plt.imshow(img, 'gray')
                cv2.imshow('num', img)
                continue
        ranksuit_list: list = list()
        for i, (img, original) in enumerate(cropped_images):
            drawable = img.copy()
            d2 = original.copy()

            contours, _ = cv2.findContours(drawable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts_sort = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            cnts_sort = sorted(cnts_sort, key=lambda x: cv2.boundingRect(x)[1])
            cv2.drawContours(drawable, cnts_sort, -1, (0, 255, 0), 1)

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
            
        black_img = np.zeros((120, 70))
        for i, ranksuit in enumerate(ranksuit_list):

            rank = black_img
            suit = black_img
            try:
                rank = ranksuit[0]
                suit = ranksuit[1]
            except:
                pass
            
         # train_ranks = Loader.load_ranks('imgs/ranks')
        train_ranks = Loader.load_ranks('imgs/ranks')
        # PokerCard/imgs/ranks
        train_suits = Loader.load_suits('imgs/suits')
        
        for it in ranksuit_list:
            try:
                rank = it[0]
                suit = it[1]
            except:
                continue
            rs = process.template_matching(rank, suit, train_ranks, train_suits)
            # print(rs)
        
        
        
        # Show the resized frame
        cv2.imshow("Image", img_resized)
        # cv2.imshow("Thresholding", imgResult2)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

        
        # PYGAME
        if not selesai_register and rs: register_card(rs)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if player_turn == "player":
                    for i, card in enumerate(player_cards):
                        if 50 <= mouse_pos[0] <= 300 and HEIGHT - 300 + i * 30 <= mouse_pos[1] <= HEIGHT - 270 + i * 30:
                            play_card(card)
                                
                        
                    if WIDTH // 2 - 75 <= mouse_pos[0] <= WIDTH // 2 + 75 and HEIGHT - 200 <= mouse_pos[1] <= HEIGHT - 150:
                        is_passed = True
                        player_turn = "ai"
                        pass_sound.play()
            if player_turn == "player" and rs and selesai_register:
                print ("player card sebelumnya: ", player_cards)
                rank, suit = rs
                card = (rank, suit)
                if play_card(card):
                    print("Card played successfully")
                    print ("player card setelahnya: ", player_cards)
                else:
                    print("Failed to play card")        

        if player_turn == "ai":
            pygame.time.wait(100)
            ai_play()

        if not player_cards and selesai_register:
            print("Player wins!")
            pygame.quit()
            sys.exit()
        elif not ai_cards:
            pygame.quit()
            sys.exit()

        game_page()
        pygame.display.flip()
        clock.tick(30)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
