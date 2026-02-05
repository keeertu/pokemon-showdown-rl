from stable_baselines3 import PPO
from pokemon_env import PokemonShowdownEnv


def main():
    print("Creating environment...")
    env = PokemonShowdownEnv()

    print("Starting PPO training...")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_logs/",

        # --- safer rollout settings ---
        n_steps=512,          # was 2048 (too slow per update)
        batch_size=64,
        n_epochs=10,

        # --- learning stability ---
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,

        # --- exploration stability ---
        ent_coef=0.01,         # encourages exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    # shorter sanity run first
    model.learn(total_timesteps=20_000)

    print("Saving model...")
    model.save("ppo_pokemon")

    print("Training complete!")


if __name__ == "__main__":
    main()
