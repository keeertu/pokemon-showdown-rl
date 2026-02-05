import time
from stable_baselines3 import PPO
from pokemon_env import PokemonShowdownEnv

def main():
    print("===========================================")
    print("  Pokemon RL Agent - Human Battle Mode")
    print("===========================================")
    print("1. Creating environment in listening mode...")
    
    # Initialize environment with human_opponent=True
    # This keeps the RL agent waiting for a challenge
    env = PokemonShowdownEnv(human_opponent=True)

    print(f"2. Loading trained model from 'ppo_pokemon.zip'...")
    try:
        model = PPO.load("ppo_pokemon")
    except FileNotFoundError:
        print("Error: 'ppo_pokemon.zip' not found. Please train the model first.")
        env.close()
        return

    print("3. Agent is ready! Please challenge the following user on your local Showdown server:")
    print(f"   Agent Username: {env.player.username}")
    print("   (Format: Gen 8 Anything Goes)")
    print("===========================================")

    # Reset environment to start listening
    obs, _ = env.reset()
    
    print("\nWaiting for challenge... (Go to http://localhost:8000 and challenge the bot)")

    done = False
    truncated = False
    total_reward = 0

    try:
        while not (done or truncated):
            # Predict action using the trained model
            # deterministic=True ensures the agent plays its best move
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Optional: Print action info if available (env prints turn info already)
            
        print(f"\nBattle finished! Total Reward: {total_reward:.2f}")
        if total_reward > 0:
            print("Agent likely WON!")
        else:
            print("Agent likely LOST!")

    except KeyboardInterrupt:
        print("\nBattle interrupted.")
    finally:
        print("Closing environment...")
        env.close()

if __name__ == "__main__":
    main()
