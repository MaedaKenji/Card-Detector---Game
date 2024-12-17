import pygame
import random
import sys
import os

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

    def draw_player_hand(self):
        # Draw human player's cards
        for i, card in enumerate(self.human_player.hand):
            x = 550 + (i * 120)
            self.screen.blit(card.image, (x, 600))

    def draw_community_cards(self):
        # Draw community cards
        for i, card in enumerate(self.community_cards):
            x = 400 + (i * 120)
            self.screen.blit(card.image, (x, 300))

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

    def play_round(self):
        # Reset and deal initial cards
        self.reset_round()
        self.deal_initial_cards()
        self.place_blinds()

        # Game loop stages
        stages = ["PRE_FLOP", "FLOP", "TURN", "RIVER"]

        for stage in stages:
            self.game_stage = stage

            # Betting rounds
            for player in self.all_players:
                if player.is_human:
                    # Wait for player action in main game loop
                    continue
                elif not player.is_folded:
                    self.bot_decision(player)

            # Deal community cards
            if stage != "PRE_FLOP":
                self.deal_community_cards()
                
    def reset_round(self):
        # Reset deck and community cards
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.game_stage = "PRE_FLOP"

        # Reset player hands and bets
        for player in self.all_players:
            player.hand = []
            player.current_bet = 0
            player.is_folded = False

    def deal_initial_cards(self):
        # Deal 2 cards to each player
        for player in self.all_players:
            player.add_card(self.deck.deal())
            player.add_card(self.deck.deal())

    def place_blinds(self):
        # Rotate blinds
        self.all_players[0].chips -= self.small_blind
        self.all_players[0].current_bet = self.small_blind

        self.all_players[1].chips -= self.big_blind
        self.all_players[1].current_bet = self.big_blind

        self.pot += self.small_blind + self.big_blind
        self.current_bet = self.big_blind

    def bot_decision(self, bot):
        """Simple bot decision-making logic"""
        actions = ['fold', 'call', 'raise']
        action = random.choice(actions)

        if action == 'fold':
            bot.is_folded = True
        elif action == 'call':
            call_amount = self.current_bet - bot.current_bet
            if bot.chips >= call_amount:
                bot.chips -= call_amount
                bot.current_bet = self.current_bet
                self.pot += call_amount
        else:  # raise
            raise_amount = random.randint(self.big_blind, self.current_bet * 2)
            if bot.chips >= raise_amount:
                bot.chips -= raise_amount
                bot.current_bet += raise_amount
                self.current_bet += raise_amount
                self.pot += raise_amount

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
        if self.game_stage == "FLOP":
            for _ in range(3):
                self.community_cards.append(self.deck.deal())
        elif self.game_stage in ["TURN", "RIVER"]:
            self.community_cards.append(self.deck.deal())


    def run(self):
        # Start first round
        self.play_round()

        # Main game loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Handle mouse clicks for actions
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.call_button.collidepoint(event.pos):
                        self.handle_player_action("CALL")
                    elif self.raise_button.collidepoint(event.pos):
                        self.handle_player_action("RAISE")
                    elif self.fold_button.collidepoint(event.pos):
                        self.handle_player_action("FOLD")

            # Drawing
            self.draw_background()
            self.draw_players()
            self.draw_player_hand()
            self.draw_community_cards()
            self.draw_buttons()
            self.draw_game_info()

            # Update display
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()
        sys.exit()


def main():
    # Create dummy avatar images if they don't exist
    def create_dummy_avatar(filename, color):
        if not os.path.exists(filename):
            avatar = pygame.Surface((200, 200))
            avatar.fill(color)
            pygame.draw.circle(avatar, WHITE, (100, 100), 80)
            pygame.image.save(avatar, filename)

    # Create dummy avatars
    create_dummy_avatar("bot_avatar1.png", (255, 0, 0))    # Red
    create_dummy_avatar("bot_avatar2.png", (0, 255, 0))   # Green
    create_dummy_avatar("bot_avatar3.png", (0, 0, 255))   # Blue
    create_dummy_avatar("bot_avatar4.png", (255, 255, 0))  # Yellow

    # Start the game
    poker_game = PokerGame()
    poker_game.run()


if __name__ == "__main__":
    main()



import pygame
import random
import sys

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

# Fonts
FONT_LARGE = pygame.font.Font(None, 48)
FONT_MEDIUM = pygame.font.Font(None, 36)
FONT_SMALL = pygame.font.Font(None, 24)

class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value
        self.image = self.load_card_image()
    
    def load_card_image(self):
        # Simplified card image generation
        card_surface = pygame.Surface((100, 150))
        card_surface.fill(WHITE)
        pygame.draw.rect(card_surface, BLACK, card_surface.get_rect(), 2)
        
        # Add value and suit
        value_text = FONT_MEDIUM.render(str(self.value), True, BLACK)
        suit_text = FONT_MEDIUM.render(self.suit, True, BLACK)
        
        card_surface.blit(value_text, (10, 10))
        card_surface.blit(suit_text, (10, 50))
        
        return card_surface
    
    def __str__(self):
        return f"{self.value}{self.suit}"

class Deck:
    def __init__(self):
        suits = ['♥', '♦', '♠', '♣']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [Card(suit, value) for suit in suits for value in values]
        random.shuffle(self.cards)
    
    def deal(self):
        return self.cards.pop()

class Player:
    def __init__(self, name, is_human=False):
        self.name = name
        self.hand = []
        self.chips = 1000
        self.current_bet = 0
        self.is_folded = False
        self.is_human = is_human
    
    def add_card(self, card):
        self.hand.append(card)

class PokerGame:
    def __init__(self):
        self.screen = SCREEN
        self.clock = pygame.time.Clock()
        
        # Players
        self.human_player = Player("Player", is_human=True)
        self.bots = [
            Player("Bot 1"),
            Player("Bot 2"),
            Player("Bot 3"),
            Player("Bot 4")
        ]
        self.all_players = [self.human_player] + self.bots
        
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
    
    def draw_background(self):
        # Poker table background
        self.screen.fill(GREEN)
        pygame.draw.ellipse(self.screen, (101, 67, 33), (100, 200, 1000, 500))
    
    def draw_player_hand(self):
        # Draw human player's cards
        for i, card in enumerate(self.human_player.hand):
            x = 550 + (i * 120)
            self.screen.blit(card.image, (x, 600))
    
    def draw_community_cards(self):
        # Draw community cards
        for i, card in enumerate(self.community_cards):
            x = 400 + (i * 120)
            self.screen.blit(card.image, (x, 300))
    
    def draw_buttons(self):
        # Draw action buttons
        pygame.draw.rect(self.screen, BLUE, self.call_button)
        pygame.draw.rect(self.screen, RED, self.raise_button)
        pygame.draw.rect(self.screen, GRAY, self.fold_button)
        
        # Button labels
        call_text = FONT_SMALL.render("Call", True, WHITE)
        raise_text = FONT_SMALL.render("Raise", True, WHITE)
        fold_text = FONT_SMALL.render("Fold", True, WHITE)
        
        self.screen.blit(call_text, (self.call_button.x + 25, self.call_button.y + 15))
        self.screen.blit(raise_text, (self.raise_button.x + 20, self.raise_button.y + 15))
        self.screen.blit(fold_text, (self.fold_button.x + 25, self.fold_button.y + 15))
    
    def draw_game_info(self):
        # Draw pot and chip information
        pot_text = FONT_MEDIUM.render(f"Pot: ${self.pot}", True, WHITE)
        chip_text = FONT_MEDIUM.render(f"Your Chips: ${self.human_player.chips}", True, WHITE)
        stage_text = FONT_MEDIUM.render(f"Stage: {self.game_stage}", True, WHITE)
        
        self.screen.blit(pot_text, (50, 50))
        self.screen.blit(chip_text, (50, 100))
        self.screen.blit(stage_text, (50, 150))
    
    def reset_round(self):
        # Reset deck and community cards
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.game_stage = "PRE_FLOP"
        
        # Reset player hands and bets
        for player in self.all_players:
            player.hand = []
            player.current_bet = 0
            player.is_folded = False
    
    def deal_initial_cards(self):
        # Deal 2 cards to each player
        for player in self.all_players:
            player.add_card(self.deck.deal())
            player.add_card(self.deck.deal())
    
    def place_blinds(self):
        # Rotate blinds 
        self.all_players[0].chips -= self.small_blind
        self.all_players[0].current_bet = self.small_blind
        
        self.all_players[1].chips -= self.big_blind
        self.all_players[1].current_bet = self.big_blind
        
        self.pot += self.small_blind + self.big_blind
        self.current_bet = self.big_blind
    
    def bot_decision(self, bot):
        """Simple bot decision-making logic"""
        actions = ['fold', 'call', 'raise']
        action = random.choice(actions)
        
        if action == 'fold':
            bot.is_folded = True
        elif action == 'call':
            call_amount = self.current_bet - bot.current_bet
            if bot.chips >= call_amount:
                bot.chips -= call_amount
                bot.current_bet = self.current_bet
                self.pot += call_amount
        else:  # raise
            raise_amount = random.randint(self.big_blind, self.current_bet * 2)
            if bot.chips >= raise_amount:
                bot.chips -= raise_amount
                bot.current_bet += raise_amount
                self.current_bet += raise_amount
                self.pot += raise_amount
    
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
        if self.game_stage == "FLOP":
            for _ in range(3):
                self.community_cards.append(self.deck.deal())
        elif self.game_stage in ["TURN", "RIVER"]:
            self.community_cards.append(self.deck.deal())
    
    def play_round(self):
        # Reset and deal initial cards
        self.reset_round()
        self.deal_initial_cards()
        self.place_blinds()
        
        # Game loop stages
        stages = ["PRE_FLOP", "FLOP", "TURN", "RIVER"]
        
        for stage in stages:
            self.game_stage = stage
            
            # Betting rounds
            for player in self.all_players:
                if player.is_human:
                    # Wait for player action in main game loop
                    continue
                elif not player.is_folded:
                    self.bot_decision(player)
            
            # Deal community cards
            if stage != "PRE_FLOP":
                self.deal_community_cards()
    
    def run(self):
        # Start first round
        self.play_round()
        
        # Main game loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Handle mouse clicks for actions
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.call_button.collidepoint(event.pos):
                        self.handle_player_action("CALL")
                    elif self.raise_button.collidepoint(event.pos):
                        self.handle_player_action("RAISE")
                    elif self.fold_button.collidepoint(event.pos):
                        self.handle_player_action("FOLD")
            
            # Drawing
            self.draw_background()
            self.draw_player_hand()
            self.draw_community_cards()
            self.draw_buttons()
            self.draw_game_info()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(30)
        
        pygame.quit()
        sys.exit()

def main():
    poker_game = PokerGame()
    poker_game.run()

if __name__ == "__main__":
    main()
    
    
import pygame
import random
import sys
import os

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
    
    def __str__(self):
        return f"{self.value}{self.suit}"

class Deck:
    def __init__(self):
        suits = ['♥', '♦', '♠', '♣']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
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
    
    def draw_player_hand(self):
        # Draw human player's cards
        for i, card in enumerate(self.human_player.hand):
            x = 550 + (i * 120)
            self.screen.blit(card.image, (x, 600))
    
    def draw_community_cards(self):
        # Draw community cards
        for i, card in enumerate(self.community_cards):
            x = 400 + (i * 120)
            self.screen.blit(card.image, (x, 300))
    
    def draw_buttons(self):
        # Draw action buttons
        pygame.draw.rect(self.screen, BLUE, self.call_button)
        pygame.draw.rect(self.screen, RED, self.raise_button)
        pygame.draw.rect(self.screen, GRAY, self.fold_button)
        
        # Button labels
        call_text = FONT_SMALL.render("Call", True, WHITE)
        raise_text = FONT_SMALL.render("Raise", True, WHITE)
        fold_text = FONT_SMALL.render("Fold", True, WHITE)
        
        self.screen.blit(call_text, (self.call_button.x + 25, self.call_button.y + 15))
        self.screen.blit(raise_text, (self.raise_button.x + 20, self.raise_button.y + 15))
        self.screen.blit(fold_text, (self.fold_button.x + 25, self.fold_button.y + 15))
    
    def draw_game_info(self):
        # Draw pot and game stage information
        pot_text = FONT_MEDIUM.render(f"Pot: ${self.pot}", True, WHITE)
        stage_text = FONT_MEDIUM.render(f"Stage: {self.game_stage}", True, WHITE)
        
        self.screen.blit(pot_text, (50, 50))
        self.screen.blit(stage_text, (50, 100))
    
    # ... [Rest of the previous methods remain the same]
    
    def run(self):
        # Start first round
        self.play_round()
        
        # Main game loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Handle mouse clicks for actions
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.call_button.collidepoint(event.pos):
                        self.handle_player_action("CALL")
                    elif self.raise_button.collidepoint(event.pos):
                        self.handle_player_action("RAISE")
                    elif self.fold_button.collidepoint(event.pos):
                        self.handle_player_action("FOLD")
            
            # Drawing
            self.draw_background()
            self.draw_players()
            self.draw_player_hand()
            self.draw_community_cards()
            self.draw_buttons()
            self.draw_game_info()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(30)
        
        pygame.quit()
        sys.exit()

def main():
    # Create dummy avatar images if they don't exist
    def create_dummy_avatar(filename, color):
        if not os.path.exists(filename):
            avatar = pygame.Surface((200, 200))
            avatar.fill(color)
            pygame.draw.circle(avatar, WHITE, (100, 100), 80)
            pygame.image.save(avatar, filename)
    
    # Create dummy avatars
    create_dummy_avatar("bot_avatar1.png", (255, 0, 0))    # Red
    create_dummy_avatar("bot_avatar2.png", (0, 255, 0))   # Green
    create_dummy_avatar("bot_avatar3.png", (0, 0, 255))   # Blue
    create_dummy_avatar("bot_avatar4.png", (255, 255, 0)) # Yellow
    
    # Start the game
    poker_game = PokerGame()
    poker_game.run()

if __name__ == "__main__":
    main()
    
    
import pygame
import random
import sys
import os

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
    
    def __str__(self):
        return f"{self.value}{self.suit}"

class Deck:
    def __init__(self):
        suits = ['♥', '♦', '♠', '♣']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
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
    
    def draw_player_hand(self):
        # Draw human player's cards
        for i, card in enumerate(self.human_player.hand):
            x = 550 + (i * 120)
            self.screen.blit(card.image, (x, 600))
    
    def draw_community_cards(self):
        # Draw community cards
        for i, card in enumerate(self.community_cards):
            x = 400 + (i * 120)
            self.screen.blit(card.image, (x, 300))
    
    def draw_buttons(self):
        # Draw action buttons
        pygame.draw.rect(self.screen, BLUE, self.call_button)
        pygame.draw.rect(self.screen, RED, self.raise_button)
        pygame.draw.rect(self.screen, GRAY, self.fold_button)
        
        # Button labels
        call_text = FONT_SMALL.render("Call", True, WHITE)
        raise_text = FONT_SMALL.render("Raise", True, WHITE)
        fold_text = FONT_SMALL.render("Fold", True, WHITE)
        
        self.screen.blit(call_text, (self.call_button.x + 25, self.call_button.y + 15))
        self.screen.blit(raise_text, (self.raise_button.x + 20, self.raise_button.y + 15))
        self.screen.blit(fold_text, (self.fold_button.x + 25, self.fold_button.y + 15))
    
    def draw_game_info(self):
        # Draw pot and game stage information
        pot_text = FONT_MEDIUM.render(f"Pot: ${self.pot}", True, WHITE)
        stage_text = FONT_MEDIUM.render(f"Stage: {self.game_stage}", True, WHITE)
        
        self.screen.blit(pot_text, (50, 50))
        self.screen.blit(stage_text, (50, 100))
    
    # ... [Rest of the previous methods remain the same]
    
    def run(self):
        # Start first round
        self.play_round()
        
        # Main game loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Handle mouse clicks for actions
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.call_button.collidepoint(event.pos):
                        self.handle_player_action("CALL")
                    elif self.raise_button.collidepoint(event.pos):
                        self.handle_player_action("RAISE")
                    elif self.fold_button.collidepoint(event.pos):
                        self.handle_player_action("FOLD")
            
            # Drawing
            self.draw_background()
            self.draw_players()
            self.draw_player_hand()
            self.draw_community_cards()
            self.draw_buttons()
            self.draw_game_info()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(30)
        
        pygame.quit()
        sys.exit()

def main():
    # Create dummy avatar images if they don't exist
    def create_dummy_avatar(filename, color):
        if not os.path.exists(filename):
            avatar = pygame.Surface((200, 200))
            avatar.fill(color)
            pygame.draw.circle(avatar, WHITE, (100, 100), 80)
            pygame.image.save(avatar, filename)
    
    # Create dummy avatars
    create_dummy_avatar("bot_avatar1.png", (255, 0, 0))    # Red
    create_dummy_avatar("bot_avatar2.png", (0, 255, 0))   # Green
    create_dummy_avatar("bot_avatar3.png", (0, 0, 255))   # Blue
    create_dummy_avatar("bot_avatar4.png", (255, 255, 0)) # Yellow
    
    # Start the game
    poker_game = PokerGame()
    poker_game.run()

if __name__ == "__main__":
    main()


    blind_placed = False
    set_initial_blinds()
