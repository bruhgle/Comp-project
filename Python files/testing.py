import random

def coin_flip():
    return random.choice(['heads', 'tails'])

def simulate_betting(starting_amount, initial_bet, num_cycles):
    current_amount = starting_amount
    current_bet = initial_bet

    for cycle in range(num_cycles):
        print(f"\nCycle {cycle + 1}:")
        print(f"Current amount: £{current_amount}")
        print(f"Current bet: £{current_bet}")

        # Perform the coin flip
        result = coin_flip()
        print(f"Coin flip result: {result}")

        # Update the amount based on the result
        if result == 'heads':
            current_amount += current_bet
            print(f"You won £{current_bet}!")
            current_bet = initial_bet  # Reset bet to £5 on winning
        else:
            current_amount -= current_bet
            print(f"You lost £{current_bet}.")
            current_bet *= 2.5  # Increase bet 2.5x on losing

            # Check if the game is over (out of money)
            if current_amount <= 0:
                print("You're out of money. Game over.")
                break

    print("\nSimulation complete.")
    print(f"Final amount: £{current_amount}")

# Set initial parameters
starting_amount = 10
initial_bet = 0.05
num_cycles = 1000  # You can change this value

# Run the simulation
simulate_betting(starting_amount, initial_bet, num_cycles)