import asyncio
import os
import time
import uuid
import gc
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from poke_env.player.player import Player
from poke_env.player import RandomPlayer
from poke_env.player.battle_order import DefaultBattleOrder, BattleOrder
from poke_env.concurrency import POKE_LOOP, create_in_poke_loop
from poke_env.ps_client import AccountConfiguration
from model_utils import build_obs

# RL Player using rendezvous synchronization
class RLPlayer(Player):
    """
    RL Player using poke-env's queue pattern for synchronization.
    choose_move() puts battle on queue and awaits order from order_queue.
    """
    
    def __init__(self, battle_format="gen8anythinggoes", team_file="ash_team.txt"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        team_path = os.path.join(base_dir, team_file)
        with open(team_path, "r") as f:
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
        """Called by poke-env to request a move."""
        self._current_battle = battle
        self._battle_finished = battle.finished
        

        if battle.finished:
            return self.choose_random_move(battle) or self.choose_default_move()
        
        # Pass battle state to the environment step()
        await self._battle_queue.put(battle)

        # Wait for the next order (step() takes action)
        try:
            # 65s timeout allows step() retry loop (60s) to control pacing
            order = await asyncio.wait_for(self._order_queue.get(), timeout=65.0)
            
            # Guard: ensure we never return None
            if order is None:
                print(f"[WARN] choose_move received None order (Turn {battle.turn}). Fallback to random.")
                return self.choose_random_move(battle) or self.choose_default_move()
                
            return order
        except asyncio.TimeoutError:
            print(f"[WARN] choose_move timeout (Turn {battle.turn}). Fallback to random.")
            return self.choose_random_move(battle) or self.choose_default_move()

    def _battle_finished_callback(self, battle):
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

    def clear_order_queue(self):
        """Clear the order queue to prevent deadlocks (called from main thread)."""
        asyncio.run_coroutine_threadsafe(
            self._drain_queue(self._order_queue), POKE_LOOP
        ).result(timeout=1.0)

    def reset_for_new_battle(self):
        self._current_battle = None
        self._battle_finished = False
        # Clear queues synchronously  
        try:
            while not self._battle_queue.empty():
                asyncio.run_coroutine_threadsafe(
                    self._drain_queue(self._battle_queue), POKE_LOOP
                ).result(timeout=1.0)
                break
        except Exception:
            pass
        try:
            while not self._order_queue.empty():
                asyncio.run_coroutine_threadsafe(
                    self._drain_queue(self._order_queue), POKE_LOOP
                ).result(timeout=1.0)
                break
        except Exception:
            pass
        # Remove only finished battles to release memory safely
        finished_tags = [tag for tag, b in self.battles.items() if b.finished]
        for tag in finished_tags:
            del self.battles[tag]

    async def _drain_queue(self, q):
        while not q.empty():
            try:
                q.get_nowait()
            except Exception:
                break


class PokemonShowdownEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, battle_format="gen8anythinggoes",
                 player_team_file="ash_team.txt",
                 opponent_team_file="leon_team.txt",
                 human_opponent=False,
                 enable_logging=False):
        super().__init__()

        self.battle_format = battle_format
        self.player_team_file = player_team_file
        self.opponent_team_file = opponent_team_file
        self.human_opponent = human_opponent
        self.enable_logging = enable_logging

        self.action_space = spaces.Discrete(10)
        # Expanded Observation Space: 48 features
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(48,),
            dtype=np.float32,
        )

        # Initialize persistent players with unique names
        self.player = RLPlayer(
            battle_format=battle_format,
            team_file=player_team_file,
        )
        
        self.opponent = None
        if not self.human_opponent:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            opp_team_path = os.path.join(base_dir, opponent_team_file)
            with open(opp_team_path, "r") as f:
                opp_team = f.read()
                
            opp_username = f"Opponent_{uuid.uuid4().hex[:8]}"
            opp_config = AccountConfiguration(opp_username, None)
            
            self.opponent = RandomPlayer(
                battle_format=battle_format,
                team=opp_team,
                account_configuration=opp_config,
            )

        self._obs = np.zeros(48, dtype=np.float32)
        self.action_mask = np.zeros(10, dtype=np.int8)
        self._current_battle = None
        self._battle_future = None

        self._prev_my_hp = None
        self._prev_opp_hp = None
        self._prev_opp_hazards = 0
        self._prev_my_boosts = 0
        self._prev_my_faints = 0
        self._prev_opp_faints = 0
        self._step_count = 0
        self._episode_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_count += 1

        # Stabilization delay
        time.sleep(0.1)
        
        # Periodic GC to keep memory low
        if self._episode_count % 20 == 0:
            gc.collect()

        # Resource tracking every 50 episodes
        if self._episode_count % 50 == 0:
            active_battles = len(self.player.battles)
            print(f"[STATS] Episode {self._episode_count} | Active Battles: {active_battles}")

        if self.enable_logging:
            print(f"\n[Episode {self._episode_count}] Resetting...")

        # Cancel previous battle task if running
        if self._battle_future is not None:
            self._battle_future.cancel()
            self._battle_future = None

        self._obs = np.zeros(48, dtype=np.float32)
        self._current_battle = None
        self._prev_my_hp = None
        self._prev_opp_hp = None
        self._prev_opp_hazards = 0
        self._prev_my_boosts = 0
        self._prev_my_faints = 0
        self._prev_opp_faints = 0
        self._step_count = 0

        self.player.reset_for_new_battle()
        # Clear finished battles to prevent leaks
        if self.opponent:
            finished_opp_tags = [tag for tag, b in self.opponent.battles.items() if b.finished]
            for tag in finished_opp_tags:
                del self.opponent.battles[tag]

        # Start next match
        max_retries = 2
        battle = None
        
        for attempt in range(max_retries + 1):
            async def start_battle():
                if self.human_opponent:
                    if self.enable_logging:
                        print(f"[INFO] RLPlayer {self.player.username} waiting for challenge...")
                    await self.player.accept_challenges(None, 1)
                else:
                    assert self.opponent is not None
                    await self.player.battle_against(self.opponent)

            if self._battle_future is not None:
                self._battle_future.cancel()

            self._battle_future = asyncio.run_coroutine_threadsafe(start_battle(), POKE_LOOP)

            # Block until first state arrives
            if self.enable_logging:
                print(f"[Episode {self._episode_count}] Waiting for battle start (Attempt {attempt + 1})...")
            
            battle = self.player.get_battle_sync(timeout=30.0)
            if battle is not None:
                break
                
            if attempt < max_retries:
                if self.enable_logging:
                    print(f"[Episode {self._episode_count}] Battle start timeout, retrying with increased delay...")
                self.player.reset_for_new_battle()
                time.sleep(0.5)

        if battle is None:
            print(f"[Episode {self._episode_count}] ERROR: Failed to start battle after {max_retries + 1} attempts")
            self.action_mask = np.ones(10, dtype=np.int8)
            return self._obs, {"action_mask": self.action_mask}

        if self.enable_logging:
            print(f"[Episode {self._episode_count}] Battle started! ID: {battle.battle_tag}")
        self._current_battle = battle
        self._obs = build_obs(battle)
        self._build_action_mask(battle)
        # Initialize reward trackers from current state
        self._prev_my_hp = self._get_team_hp(battle, my_team=True)
        self._prev_opp_hp = self._get_team_hp(battle, my_team=False)
        self._prev_opp_hazards = self._count_hazards(battle, my_side=False)
        self._prev_my_boosts = self._sum_boosts(battle, my_side=True)
        self._prev_my_faints = 0
        self._prev_opp_faints = 0

        return self._obs.copy(), {"action_mask": self.action_mask.copy()}

    def step(self, action):
        self._step_count += 1
        
        # Reward penalty for switching
        is_switch_action = (action >= 4)

        if self._current_battle is None or self._current_battle.finished:
            reward, terminated = self._compute_final_reward()
            return self._obs.copy(), reward, terminated, False, {"action_mask": self.action_mask.copy()}

        # Deadlock prevention: clear stale orders before dispatching new one
        self.player.clear_order_queue()

        # Dispatch order
        order = self._action_to_order(int(action), self._current_battle)
        if order is None:
             # Make sure we send *something* to keep the loop alive, even if it's default
             order = self.player.choose_default_move()
        self.player.put_order_sync(order)

        # Wait for turn resolution with Retry Loop
        battle = None
        # Retry for ~60 seconds (12 * 5s) to handle human thinking time or server lag
        MAX_RETRIES = 12
        for attempt in range(MAX_RETRIES):
            battle = self.player.get_battle_sync(timeout=5.0)
            if battle is not None:
                break
            
        
        if battle is None:
            # Timeout - check if battle finished
            if self._current_battle and self._current_battle.finished:
                reward, terminated = self._compute_final_reward()
                return self._obs.copy(), reward, terminated, False, {"action_mask": self.action_mask.copy()}
            
            # Real Timeout / Server Unresponsive -> Abort Episode Safely
            print(f"[WARN] Step timeout (Turn {self._current_battle.turn}) - Aborting Episode")
            # Return DONE=True to reset environment
            return self._obs.copy(), 0.0, True, False, {"action_mask": self.action_mask.copy()}

        self._current_battle = battle
        self._obs = build_obs(battle)
        self._build_action_mask(battle)

        reward, terminated = self._compute_reward(is_switch_action)
        
        if self.enable_logging:
            self._print_battle_log(action, reward)

        return self._obs.copy(), reward, terminated, False, {"action_mask": self.action_mask.copy()}

    def _print_battle_log(self, action, reward):
        battle = self._current_battle
        if battle is None: return

        active = battle.active_pokemon.species if battle.active_pokemon else "None"
        opp = battle.opponent_active_pokemon.species if battle.opponent_active_pokemon else "None"
        
        # Get action string
        action_str = "Random"
        if action < 4 and action < len(battle.available_moves):
            action_str = f"Move: {battle.available_moves[action].id}"
        elif action >= 4:
            sw_idx = action - 4
            if sw_idx < len(battle.available_switches):
                action_str = f"Switch: {battle.available_switches[sw_idx].species}"

        # Detect KOs (using current vs previous faints)
        log_event = ""
        if hasattr(self, '_prev_opp_faints') and sum(1 for mon in battle.opponent_team.values() if mon.fainted) > self._prev_opp_faints:
            log_event = " | [KO] Opponent fainted!"
        elif hasattr(self, '_prev_my_faints') and sum(1 for mon in battle.team.values() if mon.fainted) > self._prev_my_faints:
            log_event = " | [KO] My Pokémon fainted!"

        print(f"T{battle.turn:<2} | {active} vs {opp} | {action_str:<18} | Rew: {reward:+.3f}{log_event}")

    def _action_to_order(self, action, battle):
        """Convert gym action to poke-env order."""
        # Actions 0-3: moves
        # Actions 4-9: switches
        
        # Force Switch: ignore moves, ensure illegal moves don't crash the env
        if battle.force_switch:
            if action >= 4:
                switch_idx = action - 4
                if switch_idx < len(battle.available_switches):
                    return self.player.create_order(battle.available_switches[switch_idx])
            
            # Fallback for forced switch: pick first available switch if agent tried to move
            if battle.available_switches:
                return self.player.create_order(battle.available_switches[0])
            else:
                # Force switch with no switches — should not happen if masked correctly
                return self.player.choose_random_move(battle) or self.player.choose_default_move()

        # Normal move/switch logic
        if action < 4 and action < len(battle.available_moves):
            return self.player.create_order(battle.available_moves[action])
        
        if action >= 4:
            switch_idx = action - 4
            if switch_idx < len(battle.available_switches):
                return self.player.create_order(battle.available_switches[switch_idx])
        
        # Fallback to random if something goes wrong
        return self.player.choose_random_move(battle) or self.player.choose_default_move()

    # --------------------------------------------------
    # STATE HELPERS
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

    def _count_hazards(self, battle, my_side=True):
        """Count hazards on the specified side."""
        side = battle.side_conditions if my_side else battle.opponent_side_conditions
        count = 0
        if "stealthrock" in side: count += 1
        if "stickyweb" in side: count += 1
        if "spikes" in side: count += side["spikes"]
        if "toxicspikes" in side: count += side["toxicspikes"]
        return count

    def _sum_boosts(self, battle, my_side=True):
        """Sum positive boosts for active pokemon."""
        mon = battle.active_pokemon if my_side else battle.opponent_active_pokemon
        if mon is None:
            return 0
        total = 0
        for stat, val in mon.boosts.items():
            if val > 0:
                total += val
        return total

    def _compute_reward(self, is_switch_action):
        battle = self._current_battle

        if battle is None:
            return 0.0, False

        if battle.finished:
            return self._compute_final_reward()

        # -- Calculate State Differences --
        my_hp = self._get_team_hp(battle, my_team=True)
        opp_hp = self._get_team_hp(battle, my_team=False)
        opp_hazards = self._count_hazards(battle, my_side=False)
        my_boosts = self._sum_boosts(battle, my_side=True)

        if self._prev_my_hp is None:
            self._prev_my_hp = my_hp
            self._prev_opp_hp = opp_hp
            self._prev_opp_hazards = opp_hazards
            self._prev_my_boosts = my_boosts
            return 0.0, False

        # Delta calculations
        prev_my = float(self._prev_my_hp) if self._prev_my_hp is not None else 6.0
        prev_opp = float(self._prev_opp_hp) if self._prev_opp_hp is not None else 6.0

        damage_to_opp = prev_opp - opp_hp
        damage_taken = prev_my - my_hp
        
        hazards_set = max(0, opp_hazards - self._prev_opp_hazards)
        boosts_gained = max(0, my_boosts - self._prev_my_boosts)

        # Update tracking
        self._prev_my_hp = my_hp
        self._prev_opp_hp = opp_hp
        self._prev_opp_hazards = opp_hazards
        self._prev_my_boosts = my_boosts

        # Balance weights
        REWARD_DAMAGE = 1.0     
        REWARD_HAZARD = 0.1     
        REWARD_BOOST = 0.05     
        REWARD_KO = 0.5         
        PENALTY_KO = -0.5       
        PENALTY_STALL = -0.005  
        PENALTY_SWITCH = -0.08  

        # Refined Faint Tracking
        if not hasattr(self, '_prev_my_faints'): self._prev_my_faints = 0
        if not hasattr(self, '_prev_opp_faints'): self._prev_opp_faints = 0
        
        new_my_faints = sum(1 for mon in battle.team.values() if mon.fainted)
        new_opp_faints = sum(1 for mon in battle.opponent_team.values() if mon.fainted)
        
        ko_bonus = max(0, new_opp_faints - self._prev_opp_faints) * REWARD_KO
        ko_penalty = max(0, new_my_faints - self._prev_my_faints) * PENALTY_KO
        
        self._prev_my_faints = new_my_faints
        self._prev_opp_faints = new_opp_faints

        reward = 0.0
        reward += (damage_to_opp * REWARD_DAMAGE)
        reward -= (damage_taken * REWARD_DAMAGE)
        reward += (hazards_set * REWARD_HAZARD)
        reward += (boosts_gained * REWARD_BOOST)
        reward += ko_bonus
        reward += ko_penalty
        reward += PENALTY_STALL
        
        if is_switch_action:
            reward += PENALTY_SWITCH

        # Clip per-step reward (except terminal)
        reward = float(np.clip(reward, -1.0, 1.0))

        return reward, False

    def _compute_final_reward(self):
        battle = self._current_battle

        if battle is None:
            return 0.0, True
        
        # Clear state trackers
        self._prev_my_hp = None
        self._prev_my_faints = 0
        self._prev_opp_faints = 0

        if battle.won:
            print(f"[Episode {self._episode_count}] WON!")
            return 3.0, True
        elif battle.lost:
            print(f"[Episode {self._episode_count}] LOST")
            return -3.0, True
        else:
            return 0.0, True

    def _build_action_mask(self, battle):
        if battle is None:
            self.action_mask = np.ones(10, dtype=np.int8)
            return

        mask = np.zeros(10, dtype=np.int8)

        # Mask moves during forced switches
        if not battle.force_switch:
            for i in range(min(len(battle.available_moves), 4)):
                mask[i] = 1

        for i in range(min(len(battle.available_switches), 6)):
            mask[4 + i] = 1

        if mask.sum() == 0:
            # Emergency fallback: if everything is masked, allow everything (poke-env will handle legality)
            mask = np.ones(10, dtype=np.int8)

        self.action_mask = mask


    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------
    def close(self):
        if self._battle_future:
            self._battle_future.cancel()
