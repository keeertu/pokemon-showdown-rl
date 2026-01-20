import asyncio
import threading
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from poke_env.player.player import Player
from poke_env.player import RandomPlayer

# ======================================================
# Background asyncio loop (Windows-safe)
# ======================================================
_loop = asyncio.new_event_loop()

def _run_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=_run_loop, args=(_loop,), daemon=True).start()


# ======================================================
# RL Player
# ======================================================
class RLPlayer(Player):
    def __init__(self, env):
        super().__init__(battle_format="gen9randombattle")
        self.env = env
        self._pending_action = None
        self._action_event = asyncio.Event()

    async def choose_move(self, battle):
        self.env._build_obs(battle)
        self.env._build_action_mask(battle)

        await self._action_event.wait()
        self._action_event.clear()

        action = self._pending_action
        self._pending_action = None

        if action is None:
            return self.choose_random_move(battle)

        if action < len(battle.available_moves):
            return self.create_order(battle.available_moves[action])

        switch_idx = action - 4
        if 0 <= switch_idx < len(battle.available_switches):
            return self.create_order(battle.available_switches[switch_idx])

        return self.choose_random_move(battle)

    def set_action(self, action: int):
        self._pending_action = int(action)
        self._action_event.set()


# ======================================================
# Gym Environment
# ======================================================
class PokemonShowdownEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(10)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32,
        )

        self.player = RLPlayer(self)
        self._obs = None
        self.action_mask = np.zeros(10, dtype=np.int8)
        self._last_battle = None

    # --------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._obs = None

        async def start():
            opponent = RandomPlayer(battle_format="gen9randombattle")
            await self.player.battle_against(opponent)

        asyncio.run_coroutine_threadsafe(start(), _loop)

        while self._obs is None:
            time.sleep(0.01)

        return self._obs, {"action_mask": self.action_mask}

    # --------------------------------------------------
    def step(self, action):
        self.player.set_action(action)

        prev_obs = self._obs
        while self._obs is prev_obs:
            time.sleep(0.005)

        reward, terminated = self._compute_reward()
        truncated = False

        self._print_turn_info(action, reward)

        return self._obs, reward, terminated, truncated, {
            "action_mask": self.action_mask
        }

    # --------------------------------------------------
    def _compute_reward(self):
        if not self.player.battles:
            return 0.0, False

        battle = next(iter(self.player.battles.values()))
        self._last_battle = battle

        if battle.finished:
            return (1.0 if battle.won else -1.0), True

        my_hp = sum(mon.current_hp_fraction or 0 for mon in battle.team.values())
        opp_hp = sum(mon.current_hp_fraction or 0 for mon in battle.opponent_team.values())

        return my_hp - opp_hp, False

    # --------------------------------------------------
    def _build_obs(self, battle):
        type_adv = 0.0
        if battle.active_pokemon and battle.opponent_active_pokemon:
            for move in battle.available_moves:
                eff = battle.opponent_active_pokemon.damage_multiplier(move.type)
                type_adv = max(type_adv, eff / 4.0)

        obs = [
            min(battle.turn / 100, 1.0),
            *[mon.current_hp_fraction or 0.0 for mon in battle.team.values()],
            *[mon.current_hp_fraction or 0.0 for mon in battle.opponent_team.values()],
            type_adv,
            len(battle.available_moves) / 4,
            len(battle.available_switches) / 5,
        ]

        self._obs = np.array(obs, dtype=np.float32)

    # --------------------------------------------------
    def _build_action_mask(self, battle):
        mask = np.zeros(10, dtype=np.int8)

        for i in range(len(battle.available_moves)):
            mask[i] = 1

        for i in range(len(battle.available_switches)):
            mask[4 + i] = 1

        self.action_mask = mask

    # --------------------------------------------------
    def _print_turn_info(self, action, reward):
        battle = self._last_battle
        if battle is None:
            return

        active = battle.active_pokemon.species
        opp = battle.opponent_active_pokemon.species

        move_names = [m.id for m in battle.available_moves]
        switch_names = [p.species for p in battle.available_switches]

        action_str = (
            move_names[action]
            if action < len(move_names)
            else f"SWITCH->{switch_names[action - 4]}"
            if action >= 4 and (action - 4) < len(switch_names)
            else "INVALID"
        )

        print(
            f"Turn {battle.turn:>2} | "
            f"{active} vs {opp} | "
            f"Action: {action_str:<15} | "
            f"Reward: {reward:+.3f}"
        )


# ======================================================
# Sanity Test
# ======================================================
if __name__ == "__main__":
    env = PokemonShowdownEnv()
    obs, info = env.reset()

    done = False
    step = 0

    print("\n=== Battle started ===\n")

    while not done:
        valid_actions = np.where(info["action_mask"] == 1)[0]
        action = int(np.random.choice(valid_actions))

        obs, reward, done, _, info = env.step(action)
        step += 1

    print(f"\n=== Battle finished in {step} turns ===")
