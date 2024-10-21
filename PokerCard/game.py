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

# Suara
pass_sound = pygame.mixer.Sound("pass_sound.wav")
player_turn_sound = pygame.mixer.Sound("player_turn.wav")

# Rank dan Suit
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["Spades", "Hearts", "Clubs", "Diamonds"]

# Variabel game
player_turn = None  # None, "player", "ai"
player_cards = []
ai_cards = []
last_card_played = None  # (rank, suit, player)
is_passed = False  # Menyimpan status pass

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
    player_cards = deck[:10]
    ai_cards = deck[10:20]

# Fungsi untuk menentukan giliran pertama
def suit():
    return random.choice(["player", "ai"])

# Fungsi untuk login page
def login_page():
    SCREEN.fill(GREEN)
    title_text = FONT.render("Welcome to Poker Game", True, WHITE)
    SCREEN.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 4))
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
        SCREEN.blit(last_card_text, (WIDTH // 2 - last_card_text.get_width() // 2, HEIGHT // 2 - 50))

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
    if is_passed or compare_cards(card, last_card_played):
        last_card_played = (*card, "player")
        player_cards.remove(card)
        player_turn = "ai"
        is_passed = False

# Fungsi untuk membandingkan kartu
def compare_cards(card1, card2):
    if card2 is None:
        return True
    rank1, _ = card1
    rank2, _ = card2[:2]
    return RANKS.index(rank1) > RANKS.index(rank2)

# Fungsi AI bermain
def ai_play():
    global last_card_played, player_turn, is_passed
    playable_cards = [card for card in ai_cards if compare_cards(card, last_card_played)]
    if is_passed or playable_cards:
        chosen_card = random.choice(playable_cards) if playable_cards else ai_cards[0]
        last_card_played = (*chosen_card, "AI")
        ai_cards.remove(chosen_card)
        player_turn = "player"
        player_turn_sound.play()
    else:
        is_passed = True
        player_turn = "player"
        player_turn_sound.play()

# Fungsi utama
def main():
    global player_turn

    clock = pygame.time.Clock()
    deal_cards()
    player_turn = suit()

    while True:
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

        if player_turn == "ai":
            pygame.time.wait(500)
            ai_play()

        if not player_cards:
            print("Player wins!")
            pygame.quit()
            sys.exit()
        elif not ai_cards:
            print("AI wins!")
            pygame.quit()
            sys.exit()

        game_page()
        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()
