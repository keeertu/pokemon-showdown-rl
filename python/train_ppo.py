import os
from stable_baselines3 import PPO
from pokemon_env import PokemonShowdownEnv
from model_utils import get_latest_model

def main():
    model_dir = "models/ppo"
    model_prefix = "ppo_leon"
    os.makedirs(model_dir, exist_ok=True)

    print("Creating environment...")
    env = PokemonShowdownEnv(
        player_team_file="leon_team.txt",
        opponent_team_file="leon_team.txt",
    )

    latest_model_path, latest_v = get_latest_model(model_dir, model_prefix)
    
    if latest_model_path:
        print(f"Loading latest model: {latest_model_path} (Version {latest_v})")
        model = PPO.load(latest_model_path, env=env)
    else:
        print("No existing model found. Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_logs/",
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )

    print(f"Starting PPO training (Timesteps: 20,000)...")
    model.learn(total_timesteps=20_000)

    new_v = latest_v + 1
    new_model_path = os.path.join(model_dir, f"{model_prefix}_v{new_v}")
    
    print(f"Saving versioned model to {new_model_path}.zip...")
    model.save(new_model_path)

    print("Training complete!")

if __name__ == "__main__":
    main()
