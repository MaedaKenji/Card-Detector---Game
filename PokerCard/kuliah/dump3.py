import random
import time

class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value
    
    def __str__(self):
        return f"{self.value}{self.suit}"
    
    def __repr__(self):
        return self.__str__()

class Deck:
    def __init__(self):
        suits = ['♥', '♦', '♠', '♣']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [Card(suit, value) for suit in suits for value in values]
        random.shuffle(self.cards)
    
    def deal(self):
        return self.cards.pop()

class Player:
    def __init__(self, name, chips=1000):
        self.name = name
        self.hand = []
        self.chips = chips
        self.current_bet = 0
        self.is_folded = False
    
    def add_card(self, card):
        self.hand.append(card)
    
    def clear_hand(self):
        self.hand = []
        self.current_bet = 0
        self.is_folded = False

class PokerGame:
    def __init__(self, player_name):
        self.players = [Player(player_name)]
        self.bots = [
            Player("Bot 1"),
            Player("Bot 2"),
            Player("Bot 3"),
            Player("Bot 4")
        ]
        self.all_players = self.players + self.bots
        self.deck = None
        self.community_cards = []
        self.pot = 0
        self.small_blind = 10
        self.big_blind = 20
        self.current_bet = 0
    
    def reset_round(self):
        # Reset deck and community cards
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        
        # Reset player hands and bets
        for player in self.all_players:
            player.clear_hand()
    
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
    
    def show_player_hand(self):
        print("\nYour Hand:")
        for card in self.players[0].hand:
            print(card, end=" ")
        print(f"\nYour Chips: {self.players[0].chips}")
    
    def bot_decision(self, bot):
        """Simple bot decision-making logic"""
        # Simplified bot strategy
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
    
    def player_turn(self):
        """Handle player's turn with betting options"""
        while True:
            print(f"\nCurrent Pot: {self.pot}")
            print(f"Current Bet: {self.current_bet}")
            print("Actions: (c)heck/call, (r)aise, (f)old")
            action = input("Your action: ").lower()
            
            if action == 'f':
                self.players[0].is_folded = True
                print("You folded.")
                break
            
            elif action == 'c':
                call_amount = self.current_bet - self.players[0].current_bet
                if self.players[0].chips >= call_amount:
                    self.players[0].chips -= call_amount
                    self.players[0].current_bet = self.current_bet
                    self.pot += call_amount
                    print(f"You called {call_amount}")
                    break
                else:
                    print("Not enough chips to call!")
            
            elif action == 'r':
                try:
                    raise_amount = int(input("Raise amount: "))
                    if raise_amount > 0 and self.players[0].chips >= raise_amount + (self.current_bet - self.players[0].current_bet):
                        total_bet = raise_amount + (self.current_bet - self.players[0].current_bet)
                        self.players[0].chips -= total_bet
                        self.current_bet += raise_amount
                        self.players[0].current_bet = self.current_bet
                        self.pot += total_bet
                        print(f"You raised to {self.current_bet}")
                        break
                    else:
                        print("Invalid raise amount!")
                except ValueError:
                    print("Please enter a valid number!")
    
    def deal_flop(self):
        # Burn a card
        self.deck.deal()
        
        # Deal 3 community cards
        for _ in range(3):
            self.community_cards.append(self.deck.deal())
        
        print("\nFlop:")
        for card in self.community_cards:
            print(card, end=" ")
    
    def deal_turn_river(self):
        # Burn a card
        self.deck.deal()
        
        # Deal next community card
        self.community_cards.append(self.deck.deal())
        
        print("\nTurn/River:")
        print(self.community_cards[-1])
    
    def play_round(self):
        # Reset and deal
        self.reset_round()
        self.deal_initial_cards()
        
        # Show player's hand
        self.show_player_hand()
        
        # Place initial blinds
        self.place_blinds()
        
        # Pre-flop betting round
        self.player_turn()
        for bot in self.bots:
            if not bot.is_folded:
                self.bot_decision(bot)
        
        # Flop
        self.deal_flop()
        
        # Flop betting round
        self.player_turn()
        for bot in self.bots:
            if not bot.is_folded:
                self.bot_decision(bot)
        
        # Turn
        self.deal_turn_river()
        
        # Turn betting round
        self.player_turn()
        for bot in self.bots:
            if not bot.is_folded:
                self.bot_decision(bot)
        
        # River
        self.deal_turn_river()
        
        # River betting round
        self.player_turn()
        for bot in self.bots:
            if not bot.is_folded:
                self.bot_decision(bot)
        
        print(f"\nFinal Pot: {self.pot}")
    
    def start_game(self):
        print("Welcome to Texas Hold'em Poker!")
        while True:
            self.play_round()
            
            play_again = input("\nDo you want to play another round? (y/n): ").lower()
            if play_again != 'y':
                break
        
        print("Thanks for playing!")

def main():
    player_name = input("Enter your name: ")
    game = PokerGame(player_name)
    game.start_game()

if __name__ == "__main__":
    main()