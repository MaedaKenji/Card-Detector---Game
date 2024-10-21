import pygame
import sys
import random

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

# Muat efek suara
pass_sound = pygame.mixer.Sound("pass_sound.wav")
player_turn_sound = pygame.mixer.Sound("player_turn.wav")

# Variabel untuk menyimpan state game
current_page = "login"
player_turn = None  # None, "player", "ai"
player_cards = []
ai_cards = []
last_card_played = None  # Menyimpan kartu terakhir yang dimainkan
is_passed = False  # Menyimpan status pass

# Fungsi untuk menggambar tombol
def draw_button(text, x, y, width, height, color):
    pygame.draw.rect(SCREEN, color, (x, y, width, height))
    button_text = FONT.render(text, True, BLACK)
    text_rect = button_text.get_rect(center=(x + width // 2, y + height // 2))
    SCREEN.blit(button_text, text_rect)

# Fungsi untuk menggambar kartu
def draw_cards(cards, x, y):
    for i, card in enumerate(cards):
        card_text = SMALL_FONT.render(str(card), True, WHITE)
        SCREEN.blit(card_text, (x + i * 50, y))

# Fungsi untuk mengocok dan membagikan kartu
def deal_cards():
    global player_cards, ai_cards
    deck = list(range(1, 21))  # Kartu bernilai 1-20
    random.shuffle(deck)
    player_cards = deck[:10]
    ai_cards = deck[10:]

# Fungsi untuk menentukan giliran pertama dengan suit
def suit():
    return random.choice(["player", "ai"])

# Fungsi untuk login page
def login_page():
    SCREEN.fill(GREEN)

    # Judul login
    title_text = FONT.render("Welcome to Poker Game", True, WHITE)
    SCREEN.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 4))

    # Tombol play
    draw_button("Play", WIDTH // 2 - 75, HEIGHT // 2, 150, 50, YELLOW)

# Fungsi untuk halaman utama game
def game_page():
    global player_turn, last_card_played

    SCREEN.fill(GREEN)

    # Judul halaman utama
    game_text = FONT.render("Poker Game - Main Page", True, WHITE)
    SCREEN.blit(game_text, (WIDTH // 2 - game_text.get_width() // 2, 50))

    # Tampilkan kartu pemain
    draw_cards(player_cards, 50, HEIGHT - 100)

    # Tampilkan kartu terakhir yang dimainkan
    if last_card_played:
        last_card_text = SMALL_FONT.render(f"Last Played: {last_card_played}", True, WHITE)
        SCREEN.blit(last_card_text, (WIDTH // 2 - last_card_text.get_width() // 2, HEIGHT // 2 - 50))

    # Tampilkan jumlah kartu AI yang tersisa
    ai_cards_text = FONT.render(f"AI Cards Left: {len(ai_cards)}", True, WHITE)
    SCREEN.blit(ai_cards_text, (WIDTH - ai_cards_text.get_width() - 20, 20))

    # Tombol pass jika tidak ada kartu yang bisa dimainkan
    draw_button("Pass", WIDTH // 2 - 75, HEIGHT - 200, 150, 50, YELLOW)

    # Tampilkan giliran saat ini
    turn_text = FONT.render(f"Turn: {player_turn}", True, WHITE)
    SCREEN.blit(turn_text, (WIDTH // 2 - turn_text.get_width() // 2, 100))

# Fungsi untuk memainkan kartu
def play_card(card):
    global last_card_played, player_turn, is_passed
    # Jika ada pass, kartu apa pun bisa dimainkan
    if is_passed or last_card_played is None or card > last_card_played:
        last_card_played = card
        player_cards.remove(card)
        player_turn = "ai"  # Ganti giliran ke AI
        is_passed = False  # Reset status pass setelah kartu dimainkan

# Fungsi untuk AI bermain
def ai_play():
    global last_card_played, player_turn, is_passed
    # Jika pass aktif, AI bisa memainkan kartu mana pun
    if is_passed:
        chosen_card = random.choice(ai_cards)  # AI pilih kartu acak
    else:
        playable_cards = [card for card in ai_cards if last_card_played is None or card > last_card_played]
        if playable_cards:
            chosen_card = min(playable_cards)  # AI pilih kartu terendah
        else:
            is_passed = True  # AI melakukan pass
            player_turn = "player"
            player_turn_sound.play()  # Mainkan suara giliran pemain
            return  # Akhiri giliran AI

    last_card_played = chosen_card
    ai_cards.remove(chosen_card)
    player_turn = "player"  # Ganti giliran ke pemain
    player_turn_sound.play()  # Mainkan suara giliran pemain

# Fungsi utama untuk menjalankan game
def main():
    global current_page, player_turn

    clock = pygame.time.Clock()

    while True:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos

                if current_page == "login":
                    if WIDTH // 2 - 75 <= mouse_pos[0] <= WIDTH // 2 + 75 and HEIGHT // 2 <= mouse_pos[1] <= HEIGHT // 2 + 50:
                        current_page = "game"
                        deal_cards()
                        player_turn = suit()  # Tentukan giliran pertama
                elif current_page == "game":
                    if WIDTH // 2 - 75 <= mouse_pos[0] <= WIDTH // 2 + 75 and HEIGHT - 200 <= mouse_pos[1] <= HEIGHT - 150:
                        is_passed = True  # Pemain melakukan pass
                        player_turn = "ai"  # Ganti giliran ke AI
                        pass_sound.play()  # Mainkan suara saat pass
                    elif player_turn == "player":
                        for i, card in enumerate(player_cards):
                            if 50 + i * 50 <= mouse_pos[0] <= 50 + (i + 1) * 50 and HEIGHT - 100 <= mouse_pos[1] <= HEIGHT - 80:
                                play_card(card)

        # Game loop untuk menggambar halaman dan proses giliran AI
        if current_page == "login":
            login_page()
        elif current_page == "game":
            game_page()
            if player_turn == "ai":
                pygame.time.wait(500)  # AI berpikir sejenak
                ai_play()

            # Cek kondisi menang
            if not player_cards:
                print("Player wins!")
                pygame.quit()
                sys.exit()
            elif not ai_cards:
                print("AI wins!")
                pygame.quit()
                sys.exit()

        pygame.display.flip()
        clock.tick(30)

# Menjalankan game
if __name__ == "__main__":
    main()
