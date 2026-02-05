import asyncio
import threading
import time
import queue
import uuid
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from poke_env.player.player import Player
from poke_env.player import RandomPlayer
from poke_env.concurrency import POKE_LOOP, create_in_poke_loop
from poke_env.ps_client import AccountConfiguration

# Set to True to enable debug logging
DEBUG = False

# ======================================================
# RL Player with Queue-based Synchronization
# ======================================================
class RLPlayer(Player):
    """
    RL Player using poke-env's queue pattern for synchronization.
    choose_move() puts battle on queue and awaits order from order_queue.
    """
    
    def __init__(self, battle_format="gen8anythinggoes", team_file="ash_team.txt"):
        with open(team_file, "r") as f:
            team = f.read()
            
        # Generate unique username to avoid collisions/zombies
        username = f"RLPlayer_{uuid.uuid4().hex[:8]}"
        config = AccountConfiguration(username, None)
        
        super().__init__(
            battle_format=battle_format,
            team=team,
            account_configuration=config,
        )
        # Async queues for synchronization
        self._battle_queue: asyncio.Queue = create_in_poke_loop(asyncio.Queue, 1)
        self._order_queue: asyncio.Queue = create_in_poke_loop(asyncio.Queue, 1)
        self._current_battle = None
        self._battle_finished = False

    async def choose_move(self, battle):
        """
        Called by poke-env when a move decision is needed.
        Puts battle on queue, waits for order from step().
        """
        self._current_battle = battle
        self._battle_finished = battle.finished
        
        if DEBUG:
            print(f"[DEBUG] choose_move: Turn {battle.turn} | Requesting Action")

        # Put battle state on queue for step() to consume
        await self._battle_queue.put(battle)
        
        if battle.finished:
            return self.choose_random_move(battle)
        
        # Wait for step() to provide an order
        try:
            order = await asyncio.wait_for(self._order_queue.get(), timeout=60.0)
            if DEBUG:
                print(f"[DEBUG] choose_move: Turn {battle.turn} | Received Order: {order}")
            return order
        except asyncio.TimeoutError:
            print(f"[WARN] choose_move timeout (Turn {battle.turn}), using random move")
            return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        """Called when battle ends - put final state on queue."""
        self._battle_finished = True
        asyncio.run_coroutine_threadsafe(
            self._battle_queue.put(battle), POKE_LOOP
        )

    def get_battle_sync(self, timeout=30.0):
        """Sync method to get battle from queue (called from main thread)."""
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._async_get_battle(timeout), POKE_LOOP
            )
            return future.result(timeout=timeout + 5.0)
        except Exception as e:
            print(f"[ERROR] get_battle_sync: {e}")
            return None

    async def _async_get_battle(self, timeout):
        try:
            return await asyncio.wait_for(self._battle_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def put_order_sync(self, order):
        """Sync method to put order on queue (called from main thread)."""
        asyncio.run_coroutine_threadsafe(
            self._order_queue.put(order), POKE_LOOP
        )

    def reset_for_new_battle(self):
        """Reset player state for a new battle."""
        self._current_battle = None
        self._battle_finished = False
        # Clear queues synchronously  
        try:
            while not self._battle_queue.empty():
                asyncio.run_coroutine_threadsafe(
                    self._drain_queue(self._battle_queue), POKE_LOOP
                ).result(timeout=1.0)
                break
        except:
            pass
        try:
            while not self._order_queue.empty():
                asyncio.run_coroutine_threadsafe(
                    self._drain_queue(self._order_queue), POKE_LOOP
                ).result(timeout=1.0)
                break
        except:
            pass
        self.battles.clear()

    async def _drain_queue(self, q):
        while not q.empty():
            try:
                q.get_nowait()
            except:
                break


# ======================================================
# Gym Environment
# ======================================================
class PokemonShowdownEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, battle_format="gen8anythinggoes",
                 player_team_file="ash_team.txt",
                 opponent_team_file="leon_team.txt",
                 human_opponent=False):
        super().__init__()

        self.battle_format = battle_format
        self.player_team_file = player_team_file
        self.opponent_team_file = opponent_team_file
        self.human_opponent = human_opponent

        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(23,),
            dtype=np.float32,
        )

        # Initialize persistent players with unique names
        self.player = RLPlayer(
            battle_format=battle_format,
            team_file=player_team_file,
        )
        
        self.opponent = None
        if not self.human_opponent:
            with open(opponent_team_file, "r") as f:
                opp_team = f.read()
                
            opp_username = f"Opponent_{uuid.uuid4().hex[:8]}"
            opp_config = AccountConfiguration(opp_username, None)
            
            self.opponent = RandomPlayer(
                battle_format=battle_format,
                team=opp_team,
                account_configuration=opp_config,
            )

        self._obs = np.zeros(23, dtype=np.float32)
        self.action_mask = np.zeros(10, dtype=np.int8)
        self._current_battle = None
        self._battle_future = None

        self._prev_my_hp = None
        self._prev_opp_hp = None
        self._step_count = 0
        self._episode_count = 0

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_count += 1

        print(f"\n[Episode {self._episode_count}] Resetting...")

        # Cancel previous battle task if running
        if self._battle_future is not None:
            self._battle_future.cancel()
            self._battle_future = None

        self._obs = np.zeros(23, dtype=np.float32)
        self._current_battle = None
        self._prev_my_hp = None
        self._prev_opp_hp = None
        self._step_count = 0

        # Reset player state (clears queues)
        self.player.reset_for_new_battle()
        # Explicitly clear opponent battles to avoid memory leak
        if self.opponent:
            self.opponent.battles.clear()

        # Start battle asynchronously
        async def start_battle():
            if self.human_opponent:
                print(f"[INFO] RLPlayer {self.player.username} waiting for challenge...")
                await self.player.accept_challenges(None, 1)
            else:
                # Reuse persistent opponent
                await self.player.battle_against(self.opponent)

        self._battle_future = asyncio.run_coroutine_threadsafe(start_battle(), POKE_LOOP)

        # Wait for first battle state from queue
        print(f"[Episode {self._episode_count}] Waiting for battle start...")
        battle = self.player.get_battle_sync(timeout=60.0)
        
        if battle is None:
            print(f"[Episode {self._episode_count}] ERROR: Timed out waiting for battle start")
            self.action_mask = np.ones(10, dtype=np.int8)
            return self._obs, {"action_mask": self.action_mask}

        print(f"[Episode {self._episode_count}] Battle started! ID: {battle.battle_tag}")
        self._current_battle = battle
        self._build_obs(battle)
        self._build_action_mask(battle)
        self._prev_my_hp = self._get_team_hp(battle, my_team=True)
        self._prev_opp_hp = self._get_team_hp(battle, my_team=False)

        return self._obs.copy(), {"action_mask": self.action_mask.copy()}

    # --------------------------------------------------
    # STEP
    # --------------------------------------------------
    def step(self, action):
        self._step_count += 1

        if self._current_battle is None or self._current_battle.finished:
            reward, terminated = self._compute_final_reward()
            return self._obs.copy(), reward, terminated, False, {"action_mask": self.action_mask.copy()}

        # Convert action to order and send to player
        order = self._action_to_order(int(action), self._current_battle)
        self.player.put_order_sync(order)

        # Wait for next battle state (blocks until turn advances or battle ends)
        battle = self.player.get_battle_sync(timeout=30.0)

        if battle is None:
            # Timeout - check if battle finished
            if self._current_battle and self._current_battle.finished:
                reward, terminated = self._compute_final_reward()
                return self._obs.copy(), reward, terminated, False, {"action_mask": self.action_mask.copy()}
            # Return current state with small penalty
            print(f"[WARN] Step timeout at turn {self._current_battle.turn}")
            return self._obs.copy(), -0.02, False, False, {"action_mask": self.action_mask.copy()}

        self._current_battle = battle
        self._build_obs(battle)
        self._build_action_mask(battle)

        reward, terminated = self._compute_reward()
        
        if DEBUG:
            self._print_turn_info(action, reward)

        return self._obs.copy(), reward, terminated, False, {"action_mask": self.action_mask.copy()}

    # --------------------------------------------------
    # ACTION TO ORDER
    # --------------------------------------------------
    def _action_to_order(self, action, battle):
        """Convert discrete action to poke-env order."""
        # Actions 0-3: moves
        # Actions 4-9: switches
        
        if action < 4 and action < len(battle.available_moves):
            return self.player.create_order(battle.available_moves[action])
        
        if action >= 4:
            switch_idx = action - 4
            if switch_idx < len(battle.available_switches):
                return self.player.create_order(battle.available_switches[switch_idx])
        
        # Fallback to random valid move
        return self.player.choose_random_move(battle)

    # --------------------------------------------------
    # BATTLE STATUS HELPERS
    # --------------------------------------------------
    def _get_team_hp(self, battle, my_team=True):
        if battle is None:
            return 0.0
        team = battle.team if my_team else battle.opponent_team
        total = 0.0
        for mon in team.values():
            hp_frac = mon.current_hp_fraction
            if hp_frac is not None:
                total += float(hp_frac)
            elif not mon.fainted:
                total += 1.0
        return total

    # --------------------------------------------------
    # REWARD
    # --------------------------------------------------
    def _compute_reward(self):
        battle = self._current_battle

        if battle is None:
            return 0.0, False

        if battle.finished:
            return self._compute_final_reward()

        my_hp = self._get_team_hp(battle, my_team=True)
        opp_hp = self._get_team_hp(battle, my_team=False)

        if self._prev_my_hp is None:
            self._prev_my_hp = my_hp
            self._prev_opp_hp = opp_hp
            return 0.0, False

        prev_my = float(self._prev_my_hp) if self._prev_my_hp is not None else 0.0
        prev_opp = float(self._prev_opp_hp) if self._prev_opp_hp is not None else 0.0

        damage_to_opp = prev_opp - opp_hp
        damage_taken = prev_my - my_hp

        self._prev_my_hp = my_hp
        self._prev_opp_hp = opp_hp

        my_faints = sum(1 for mon in battle.team.values() if mon.fainted)
        opp_faints = sum(1 for mon in battle.opponent_team.values() if mon.fainted)

        reward = damage_to_opp - damage_taken + 0.5 * (opp_faints - my_faints)
        reward -= 0.02
        reward *= 1.2
        reward = float(np.clip(reward, -1.0, 1.0))

        return reward, False

    def _compute_final_reward(self):
        battle = self._current_battle

        if battle is None:
            return 0.0, True

        if battle.won:
            print(f"[Episode {self._episode_count}] WON!")
            return 2.0, True
        elif battle.lost:
            print(f"[Episode {self._episode_count}] LOST")
            return -2.0, True
        else:
            return 0.0, True

    # --------------------------------------------------
    # OBSERVATION
    # --------------------------------------------------
    def _build_obs(self, battle):
        if battle is None:
            return

        type_adv = 0.0

        if battle.active_pokemon and battle.opponent_active_pokemon:
            for move in battle.available_moves:
                try:
                    eff = battle.opponent_active_pokemon.damage_multiplier(move.type)
                    type_adv = max(type_adv, eff / 4.0)
                except:
                    pass

        my_hp = []
        for mon in battle.team.values():
            hp_frac = mon.current_hp_fraction
            if hp_frac is not None:
                my_hp.append(float(hp_frac))
            elif mon.fainted:
                my_hp.append(0.0)
            else:
                my_hp.append(1.0)

        opp_hp = []
        for mon in battle.opponent_team.values():
            hp_frac = mon.current_hp_fraction
            if hp_frac is not None:
                opp_hp.append(float(hp_frac))
            elif mon.fainted:
                opp_hp.append(0.0)
            else:
                opp_hp.append(1.0)

        my_hp = (my_hp + [0.0] * 6)[:6]
        opp_hp = (opp_hp + [0.0] * 6)[:6]

        my_faints = sum(1 for mon in battle.team.values() if mon.fainted) / 6.0
        opp_faints = sum(1 for mon in battle.opponent_team.values() if mon.fainted) / 6.0

        my_status = 0.0
        if battle.active_pokemon is not None:
            my_status = float(battle.active_pokemon.status is not None)

        opp_status = 0.0
        if battle.opponent_active_pokemon is not None:
            opp_status = float(battle.opponent_active_pokemon.status is not None)

        weather_flag = float(battle.weather is not None and len(battle.weather) > 0)
        terrain_flag = float(battle.fields is not None and len(battle.fields) > 0)

        turn_norm = min(battle.turn / 100.0, 1.0)
        moves_norm = len(battle.available_moves) / 4.0
        switches_norm = len(battle.available_switches) / 5.0

        obs = [
            turn_norm,
            *my_hp,
            *opp_hp,
            type_adv,
            moves_norm,
            switches_norm,
            my_faints,
            opp_faints,
            my_status,
            opp_status,
            weather_flag,
            terrain_flag,
        ]

        arr = np.array(obs, dtype=np.float32)

        if arr.shape[0] < 23:
            arr = np.pad(arr, (0, 23 - arr.shape[0]))
        elif arr.shape[0] > 23:
            arr = arr[:23]

        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

        self._obs = arr

    # --------------------------------------------------
    # ACTION MASK
    # --------------------------------------------------
    def _build_action_mask(self, battle):
        if battle is None:
            self.action_mask = np.ones(10, dtype=np.int8)
            return

        mask = np.zeros(10, dtype=np.int8)

        for i in range(min(len(battle.available_moves), 4)):
            mask[i] = 1

        for i in range(min(len(battle.available_switches), 6)):
            mask[4 + i] = 1

        if mask.sum() == 0:
            mask = np.ones(10, dtype=np.int8)

        self.action_mask = mask

    # --------------------------------------------------
    # DEBUG PRINT
    # --------------------------------------------------
    def _print_turn_info(self, action, reward):
        battle = self._current_battle

        if battle is None:
            return
        if battle.active_pokemon is None:
            return
        if battle.opponent_active_pokemon is None:
            return

        active = battle.active_pokemon.species
        opp = battle.opponent_active_pokemon.species

        move_names = [m.id for m in battle.available_moves]
        switch_names = [p.species for p in battle.available_switches]

        action_str = "AUTO"

        if isinstance(action, (int, np.integer)):
            if 0 <= action < len(move_names):
                action_str = move_names[action]
            elif action >= 4 and (action - 4) < len(switch_names):
                action_str = f"SWITCH->{switch_names[action - 4]}"

        print(
            f"Turn {battle.turn:>2} | "
            f"{active} vs {opp} | "
            f"Action: {action_str:<15} | "
            f"Reward: {reward:+.3f}"
        )

    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------
    def close(self):
        if self._battle_future:
            self._battle_future.cancel()
