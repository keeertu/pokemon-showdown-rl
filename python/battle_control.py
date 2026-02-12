import asyncio
import os
import numpy as np
import uuid
from stable_baselines3 import PPO, DQN
from poke_env.player import Player
from poke_env import AccountConfiguration
from model_utils import get_latest_model, build_obs

class RLModelPlayer(Player):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def choose_move(self, battle):
        obs = build_obs(battle)
        action, _ = self.model.predict(obs, deterministic=True)


        # 1. FORCED SWITCH: only switches are legal
        if battle.force_switch:
            if action >= 4:
                switch_idx = action - 4
                if switch_idx < len(battle.available_switches):
                    return self.create_order(battle.available_switches[switch_idx])
            # Fallback: pick first available switch
            if battle.available_switches:
                return self.create_order(battle.available_switches[0])
            # No switches available (should not happen), let server decide
            return self.choose_default_move()

        # 2. NORMAL: try move
        if action < 4 and action < len(battle.available_moves):
            return self.create_order(battle.available_moves[action])

        # 3. NORMAL: try switch
        if action >= 4:
            switch_idx = action - 4
            if switch_idx < len(battle.available_switches):
                return self.create_order(battle.available_switches[switch_idx])

        # 4. CATCH-ALL: model picked something invalid, use random legal move
        return self.choose_random_move(battle)

async def main():
    # Setup paths
    ppo_dir, ppo_prefix = "models/ppo", "ppo_leon"
    dqn_dir, dqn_prefix = "models/dqn", "dqn_leon"
    
    ppo_path, ppo_v = get_latest_model(ppo_dir, ppo_prefix)
    dqn_path, dqn_v = get_latest_model(dqn_dir, dqn_prefix)
    
    if not ppo_path or not dqn_path:
        print("Error: Could not find both PPO and DQN models. Please train them first.")
        return

    print(f"Loading PPO (Ash) version {ppo_v} from '{ppo_path}'...")
    ppo_model = PPO.load(ppo_path)
    
    print(f"Loading DQN (Ash) version {dqn_v} from '{dqn_path}'...")
    dqn_model = DQN.load(dqn_path)
    
    # Load team and create players
    base_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_path, "ash_team.txt"), "r") as f:
        ash_team = f.read()

    ppo_player = RLModelPlayer(
        model=ppo_model,
        battle_format="gen8anythinggoes",
        team=ash_team,
        account_configuration=AccountConfiguration(f"PPO_Ash_C_{uuid.uuid4().hex[:4]}", None)
    )
    
    dqn_player = RLModelPlayer(
        model=dqn_model,
        battle_format="gen8anythinggoes",
        team=ash_team,
        account_configuration=AccountConfiguration(f"DQN_Ash_C_{uuid.uuid4().hex[:4]}", None)
    )

    # Start evaluation
    n_battles = 100
    print(f"\n--- Starting {n_battles} Battles (Control): PPO (v{ppo_v}) vs DQN (v{dqn_v}) ---\n")
    
    await ppo_player.battle_against(dqn_player, n_battles=n_battles)

    # Report results
    ppo_wins = ppo_player.n_won_battles
    dqn_wins = dqn_player.n_won_battles
    
    print("\n--- Control Experiment Results ---")
    print(f"Total Battles: {n_battles}")
    print(f"PPO (Ash) Wins: {ppo_wins}")
    print(f"DQN (Ash) Wins: {dqn_wins}")
    print(f"PPO Win Rate: {(ppo_wins/n_battles)*100:.2f}%")
    print(f"DQN Win Rate: {(dqn_wins/n_battles)*100:.2f}%")
    print("----------------------------")

if __name__ == "__main__":
    asyncio.run(main())
