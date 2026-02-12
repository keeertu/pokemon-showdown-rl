import os
from stable_baselines3 import PPO, DQN
from pokemon_env import PokemonShowdownEnv
from model_utils import get_latest_model

def main():
    print("  Battle Agent - Human Mode")
    print("===========================================")
    
    print("Select Agent:")
    print("   [1] PPO (Leon's Team)")
    print("   [2] DQN (Leon's Team)")
    
    choice = input("\nEnter choice (1 or 2, default 1): ").strip() or "1"
    
    if choice == "1":
        model_dir = "models/ppo"
        model_prefix = "ppo_leon"
        ModeClass = PPO
        mode_name = "PPO"
    else:
        model_dir = "models/dqn"
        model_prefix = "dqn_leon"
        ModeClass = DQN
        mode_name = "DQN"

    print(f"\nSearching for latest {mode_name} model...")
    model_path, latest_v = get_latest_model(model_dir, model_prefix)
    
    if not model_path:
        print(f"Error: No {mode_name} models found in '{model_dir}'. Please train the model first.")
        return

    print(f"Loading version {latest_v}...")
    try:
        model = ModeClass.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Initializing listener...")
    env = PokemonShowdownEnv(
        player_team_file="leon_team.txt",
        human_opponent=True,
    )

    print(f"Agent ({mode_name}) is active. Challenge it on your local Showdown server.")
    print(f"Username: {env.player.username}")
    print("Format: Gen 8 Anything Goes")
    print("===========================================")

    # Listen for challenges
    obs, _ = env.reset()
    
    print("\nWaiting for challenge... (Go to http://localhost:8000 and challenge the bot)")

    done = False
    truncated = False
    total_reward = 0

    try:
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
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
