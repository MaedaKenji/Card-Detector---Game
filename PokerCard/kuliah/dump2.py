import pygame
import random
import sys
import os
import time

# Initialize Pygame
pygame.init()
pygame.font.init()

# Screen Dimensions
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Texas Hold'em Poker")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
RED = (220, 20, 60)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)

# Fonts
FONT_LARGE = pygame.font.Font(None, 48)
FONT_MEDIUM = pygame.font.Font(None, 36)
FONT_SMALL = pygame.font.Font(None, 24)


class BotAvatar:
    def __init__(self, name, image_path):
        self.name = name
        # Load and scale avatar
        original_image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(original_image, (100, 100))
        self.rect = self.image.get_rect()


class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value
        self.image = self.load_card_image()
        self.back_image = self.load_back_image()

    def load_card_image(self):
        # Simplified card image generation
        card_surface = pygame.Surface((100, 150), pygame.SRCALPHA)
        card_surface.fill((255, 255, 255, 230))
        pygame.draw.rect(card_surface, BLACK, card_surface.get_rect(), 2)

        # Add value and suit
        value_text = FONT_MEDIUM.render(str(self.value), True, BLACK)
        suit_text = FONT_MEDIUM.render(self.suit, True, BLACK)

        card_surface.blit(value_text, (10, 10))
        card_surface.blit(suit_text, (10, 50))

        return card_surface

    def load_back_image(self):
        # Create a card back image
        back_surface = pygame.Surface((100, 150), pygame.SRCALPHA)
        back_surface.fill((100, 100, 200, 230))
        pygame.draw.rect(back_surface, BLACK, back_surface.get_rect(), 2)

        # Add "POKER" text
        poker_text = FONT_SMALL.render("POKER", True, WHITE)
        text_rect = poker_text.get_rect(center=(50, 75))
        back_surface.blit(poker_text, text_rect)

        return back_surface

    def __str__(self):
        return f"{self.value}{self.suit}"


class Deck:
    def __init__(self):
        suits = ['♥', '♦', '♠', '♣']
        values = ['2', '3', '4', '5', '6', '7',
                  '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [Card(suit, value) for suit in suits for value in values]
        random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop()


class Player:
    def __init__(self, name, is_human=False, avatar=None):
        self.name = name
        self.hand = []
        self.chips = 1000
        self.current_bet = 0
        self.is_folded = False
        self.is_human = is_human
        self.avatar = avatar

    def add_card(self, card):
        self.hand.append(card)


class PokerGame:
    def __init__(self):
        self.screen = SCREEN
        self.clock = pygame.time.Clock()

        # Load bot avatars
        bot_avatars = [
            BotAvatar("Bot 1", "bot_avatar1.png"),
            BotAvatar("Bot 2", "bot_avatar2.png"),
            BotAvatar("Bot 3", "bot_avatar3.png"),
            BotAvatar("Bot 4", "bot_avatar4.png")
        ]

        # Players with avatars
        self.human_player = Player("Player", is_human=True)
        self.bots = [
            Player("Bot 1", avatar=bot_avatars[0]),
            Player("Bot 2", avatar=bot_avatars[1]),
            Player("Bot 3", avatar=bot_avatars[2]),
            Player("Bot 4", avatar=bot_avatars[3])
        ]
        self.all_players = [self.human_player] + self.bots

        # Player positions on screen
        self.player_positions = [
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 200),  # Human player
            (100, 300),  # Bot 1 (left)
            (300, 100),  # Bot 2 (top left)
            (900, 100),  # Bot 3 (top right)
            (1100, 300)  # Bot 4 (right)
        ]

        # Dealer button
        self.dealer_index = 0
        self.dealer_button_image = self.create_dealer_button()

        # Game variables
        self.deck = None
        self.community_cards = []
        self.pot = 0
        self.small_blind = 10
        self.big_blind = 20
        self.current_bet = 0

        # Button rects
        self.call_button = pygame.Rect(400, 700, 100, 50)
        self.raise_button = pygame.Rect(550, 700, 100, 50)
        self.fold_button = pygame.Rect(700, 700, 100, 50)

        # Game state
        self.game_stage = "PRE_FLOP"
        self.current_player_index = 0

    def create_dealer_button(self):
        # Create dealer button surface
        button = pygame.Surface((50, 50), pygame.SRCALPHA)
        pygame.draw.circle(button, YELLOW, (25, 25), 25)

        # Add 'D' text
        d_text = FONT_MEDIUM.render("D", True, BLACK)
        text_rect = d_text.get_rect(center=(25, 25))
        button.blit(d_text, text_rect)

        return button

    def draw_background(self):
        # Poker table background with texture
        self.screen.fill(GREEN)

        # Create a slightly darker ellipse to represent the table
        pygame.draw.ellipse(self.screen, (20, 100, 50), (100, 200, 1000, 500))
        pygame.draw.ellipse(self.screen, (30, 120, 60), (120, 220, 960, 460))

    def draw_players(self):
            # Draw player avatars and their info
        for i, player in enumerate(self.all_players):
            # Position
            x, y = self.player_positions[i]

            # Draw avatar if bot has an avatar
            if player.avatar:
                avatar_rect = player.avatar.image.get_rect(center=(x, y))
                self.screen.blit(player.avatar.image, avatar_rect)

            # Player name
            name_text = FONT_SMALL.render(player.name, True, WHITE)
            name_rect = name_text.get_rect(center=(x, y + 60))
            self.screen.blit(name_text, name_rect)

            # Chips
            chips_text = FONT_SMALL.render(f"${player.chips}", True, WHITE)
            chips_rect = chips_text.get_rect(center=(x, y + 80))
            self.screen.blit(chips_text, chips_rect)

            # Highlight current player
            if i == self.current_player_index:
                pygame.draw.rect(self.screen, RED, (x-60, y-60, 120, 120), 3)

        # Draw dealer button
        dealer_x, dealer_y = self.player_positions[self.current_player_index]
        dealer_button_rect = self.dealer_button_image.get_rect(
            center=(dealer_x + 50, dealer_y - 50))
        self.screen.blit(self.dealer_button_image, dealer_button_rect)

    def update_screen(self):
        self.draw_background()
        self.draw_players()
        self.draw_player_hand()
        self.draw_community_cards()
        self.draw_buttons()
        self.draw_game_info()
        pygame.display.flip()

    
    
    def draw_player_hand(self):
        # Draw human player's cards
        for i, card in enumerate(self.human_player.hand):
            x = 550 + (i * 120)
            self.screen.blit(card.image, (x, 600))

    def draw_community_cards(self):
        # Draw community cards
        for i, card in enumerate(self.community_cards):
            x = 400 + (i * 120)
            # Use back image if not revealed yet
            card_image = card.image if i < self.community_cards_revealed else card.back_image
            self.screen.blit(card_image, (x, 300))

    def draw_buttons(self):
        # Draw action buttons
        pygame.draw.rect(self.screen, BLUE, self.call_button)
        pygame.draw.rect(self.screen, RED, self.raise_button)
        pygame.draw.rect(self.screen, GRAY, self.fold_button)

        # Button labels
        call_text = FONT_SMALL.render("Call", True, WHITE)
        raise_text = FONT_SMALL.render("Raise", True, WHITE)
        fold_text = FONT_SMALL.render("Fold", True, WHITE)

        self.screen.blit(call_text, (self.call_button.x +
                         25, self.call_button.y + 15))
        self.screen.blit(raise_text, (self.raise_button.x +
                         20, self.raise_button.y + 15))
        self.screen.blit(fold_text, (self.fold_button.x +
                         25, self.fold_button.y + 15))

    def draw_game_info(self):
        # Draw pot and game stage information
        pot_text = FONT_MEDIUM.render(f"Pot: ${self.pot}", True, WHITE)
        stage_text = FONT_MEDIUM.render(
            f"Stage: {self.game_stage}", True, WHITE)

        self.screen.blit(pot_text, (50, 50))
        self.screen.blit(stage_text, (50, 100))

    def reset_round(self):
        # Reset deck and community cards
        self.deck = Deck()
        self.community_cards = []
        self.community_cards_revealed = 0
        self.pot = 0
        self.current_bet = 0
        self.game_stage = "PRE_FLOP"

        # Reset player hands and bets
        for player in self.all_players:
            player.hand = []
            player.current_bet = 0
            player.is_folded = False

        # Move dealer button
        self.dealer_index = (self.dealer_index + 1) % len(self.all_players)
        # Set current player to after the big blind
        self.current_player_index = (
            self.dealer_index + 3) % len(self.all_players)

    def deal_initial_cards(self):
        # Deal 2 cards to each player
        for player in self.all_players:
            player.add_card(self.deck.deal())
            player.add_card(self.deck.deal())

    def place_blinds(self):
        # Rotate blinds
        small_blind_player = (self.dealer_index + 1) % len(self.all_players)
        big_blind_player = (self.dealer_index + 2) % len(self.all_players)

        self.all_players[small_blind_player].chips -= self.small_blind
        self.all_players[small_blind_player].current_bet = self.small_blind

        self.all_players[big_blind_player].chips -= self.big_blind
        self.all_players[big_blind_player].current_bet = self.big_blind

        self.pot += self.small_blind + self.big_blind
        self.current_bet = self.big_blind

    def bot_decision(self, bot):
        """Simple bot decision-making logic"""
        # Simulate thinking time
        pygame.time.delay(2000)  # 2-second delay

        actions = ['fold', 'call', 'raise']
        action = random.choice(actions)

        if action == 'fold':
            bot.is_folded = True
            print(f"{bot.name} folds.")
        elif action == 'call':
            call_amount = self.current_bet - bot.current_bet
            if bot.chips >= call_amount:
                bot.chips -= call_amount
                bot.current_bet = self.current_bet
                self.pot += call_amount
                print(f"{bot.name} calls {call_amount}")
        else:  # raise
            raise_amount = random.randint(self.big_blind, self.current_bet * 2)
            if bot.chips >= raise_amount:
                bot.chips -= raise_amount
                bot.current_bet += raise_amount
                self.current_bet += raise_amount
                self.pot += raise_amount
                print(f"{bot.name} raises to {self.current_bet}")

    def handle_player_action(self, action):
        if action == "CALL":
            call_amount = self.current_bet - self.human_player.current_bet
            if self.human_player.chips >= call_amount:
                self.human_player.chips -= call_amount
                self.human_player.current_bet = self.current_bet
                self.pot += call_amount
        elif action == "FOLD":
            self.human_player.is_folded = True
        elif action == "RAISE":
            # Simplified raise - could be expanded with input mechanism
            raise_amount = self.big_blind * 2
            if self.human_player.chips >= raise_amount:
                self.human_player.chips -= raise_amount
                self.current_bet += raise_amount
                self.human_player.current_bet = self.current_bet
                self.pot += raise_amount

    def deal_community_cards(self):
        # Burn a card
        self.deck.deal()

        # Deal community cards based on stage
        if self.game_stage == "FLOP" and self.community_cards_revealed < 3:
            for _ in range(3):
                self.community_cards.append(self.deck.deal())
            self.community_cards_revealed = 3
        elif self.game_stage == "TURN" and self.community_cards_revealed < 4:
            self.community_cards.append(self.deck.deal())
            self.community_cards_revealed = 4
        elif self.game_stage == "RIVER" and self.community_cards_revealed < 5:
            self.community_cards.append(self.deck.deal())
            self.community_cards_revealed = 5

    def play_betting_round(self):
        # Play a betting round for each player
        for _ in range(len(self.all_players)):
            current_player = self.all_players[self.current_player_index]

            if current_player.is_folded:
                self.current_player_index = (
                    self.current_player_index + 1) % len(self.all_players)
                continue  # Skip folded players

            if current_player.is_human:
                # Handle human player's action
                action = None
                while action is None:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if self.call_button.collidepoint(event.pos):
                                action = "CALL"
                            elif self.raise_button.collidepoint(event.pos):
                                action = "RAISE"
                            elif self.fold_button.collidepoint(event.pos):
                                action = "FOLD"

                self.handle_player_action(action)
            else:
                # Handle bot's action
                self.bot_decision(current_player)

            self.current_player_index = (
                self.current_player_index + 1) % len(self.all_players)

    def main_loop(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.draw_background()
            self.draw_players()
            self.draw_player_hand()
            self.draw_community_cards()
            self.draw_buttons()
            self.draw_game_info()

            if self.game_stage == "PRE_FLOP":
                self.reset_round()
                self.deal_initial_cards()
                self.place_blinds()
                self.game_stage = "FLOP"
            elif self.game_stage == "FLOP":
                self.deal_community_cards()
                self.play_betting_round()
                self.game_stage = "TURN"
            elif self.game_stage == "TURN":
                self.deal_community_cards()
                self.play_betting_round()
                self.game_stage = "RIVER"
            elif self.game_stage == "RIVER":
                self.deal_community_cards()
                self.play_betting_round()
                self.game_stage = "SHOWDOWN"  # Add logic for showdown here

            pygame.display.flip()
            self.clock.tick(30)


if __name__ == "__main__":
    game = PokerGame()
    game.main_loop()
