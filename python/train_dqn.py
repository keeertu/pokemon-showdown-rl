import os
from stable_baselines3 import DQN
from pokemon_env import PokemonShowdownEnv
from model_utils import get_latest_model

def main():
    model_dir = "models/dqn"
    model_prefix = "dqn_leon"
    os.makedirs(model_dir, exist_ok=True)

    print("Creating environment...")
    env = PokemonShowdownEnv(
        player_team_file="leon_team.txt",
        opponent_team_file="leon_team.txt",
    )

    latest_model_path, latest_v = get_latest_model(model_dir, model_prefix)
    
    if latest_model_path:
        print(f"Loading latest model: {latest_model_path} (Version {latest_v})")
        model = DQN.load(latest_model_path, env=env)
    else:
        print("No existing model found. Creating new DQN model...")
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./dqn_logs/",
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
        )

    print(f"Starting DQN training (Timesteps: 20,000)...")
    model.learn(total_timesteps=20_000)

    new_v = latest_v + 1
    new_model_path = os.path.join(model_dir, f"{model_prefix}_v{new_v}")
    
    print(f"Saving versioned model to {new_model_path}.zip...")
    model.save(new_model_path)

    print("Training complete!")

if __name__ == "__main__":
    main()
