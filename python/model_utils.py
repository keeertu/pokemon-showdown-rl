import os
import re
import numpy as np

def get_latest_model(folder, prefix):
    """Scans the folder for versioned models (prefix_vX.zip) and returns the latest path."""
    if not os.path.exists(folder):
        return None, 0
    
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".zip")]
    if not files:
        return None, 0
    
    versions = []
    for f in files:
        match = re.search(r'_v(\d+)\.zip$', f)
        if match:
            versions.append(int(match.group(1)))
    
    if not versions:
        # Check for non-versioned legacy if any (optional safety)
        if os.path.exists(os.path.join(folder, f"{prefix}.zip")):
            return os.path.join(folder, f"{prefix}.zip"), 0
        return None, 0
    
    latest_v = max(versions)
    latest_file = os.path.join(folder, f"{prefix}_v{latest_v}.zip")
    return latest_file, latest_v

def build_obs(battle):
    """
    Consolidated observation builder (48 features).
    Ensures consistency between training and inference.
    """
    if battle is None:
        return np.zeros(48, dtype=np.float32)

    # 0: Turn (Normalized 0-1)
    turn_norm = min(battle.turn / 100.0, 1.0)

    # 1-6: My Team HP
    my_hp = []
    for mon in battle.team.values():
        hp_frac = mon.current_hp_fraction
        if hp_frac is not None: my_hp.append(float(hp_frac))
        elif mon.fainted: my_hp.append(0.0)
        else: my_hp.append(1.0)
    my_hp = (my_hp + [0.0] * 6)[:6]

    # 7-12: Opp Team HP
    opp_hp = []
    for mon in battle.opponent_team.values():
        hp_frac = mon.current_hp_fraction
        if hp_frac is not None: opp_hp.append(float(hp_frac))
        elif mon.fainted: opp_hp.append(0.0)
        else: opp_hp.append(1.0)
    opp_hp = (opp_hp + [0.0] * 6)[:6]

    # 13-17: My Boosts (atk, def, spa, spd, spe)
    my_boosts = [0.5] * 5
    if battle.active_pokemon:
        b = battle.active_pokemon.boosts
        my_boosts = [(b.get(s, 0) + 6) / 12.0 for s in ['atk', 'def', 'spa', 'spd', 'spe']]

    # 18-22: Opp Boosts
    opp_boosts = [0.5] * 5
    if battle.opponent_active_pokemon:
        b = battle.opponent_active_pokemon.boosts
        opp_boosts = [(b.get(s, 0) + 6) / 12.0 for s in ['atk', 'def', 'spa', 'spd', 'spe']]

    # 23-26: My Hazards
    my_side = battle.side_conditions
    my_hazards = [
        1.0 if 'stealthrock' in my_side else 0.0,
        min(my_side.get('spikes', 0) / 3.0, 1.0),
        min(my_side.get('toxicspikes', 0) / 2.0, 1.0),
        1.0 if 'stickyweb' in my_side else 0.0,
    ]

    # 27-30: Opp Hazards
    opp_side = battle.opponent_side_conditions
    opp_hazards = [
        1.0 if 'stealthrock' in opp_side else 0.0,
        min(opp_side.get('spikes', 0) / 3.0, 1.0),
        min(opp_side.get('toxicspikes', 0) / 2.0, 1.0),
        1.0 if 'stickyweb' in opp_side else 0.0,
    ]

    # 31: Speed Advantage
    speed_adv = 0.5
    if battle.active_pokemon and battle.opponent_active_pokemon:
        my_spe = battle.active_pokemon.base_stats.get('spe', 100)
        opp_spe = battle.opponent_active_pokemon.base_stats.get('spe', 100)
        m_b = battle.active_pokemon.boosts.get('spe', 0)
        o_b = battle.opponent_active_pokemon.boosts.get('spe', 0)
        my_mult = (m_b + 2) / 2 if m_b >= 0 else 2 / (abs(m_b) + 2)
        opp_mult = (o_b + 2) / 2 if o_b >= 0 else 2 / (abs(o_b) + 2)
        my_final = my_spe * my_mult
        opp_final = opp_spe * opp_mult
        total = my_final + opp_final
        if total > 0.001:
            speed_adv = my_final / total

    # 32-43: Move Features (4 moves * 3 features)
    move_feats = []
    for i in range(4):
        if i < len(battle.available_moves):
            move = battle.available_moves[i]
            pow_val = move.base_power / 150.0
            eff_val = 1.0
            if battle.opponent_active_pokemon:
                eff_val = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=battle.opponent_active_pokemon._data.type_chart
                )
            eff_norm = min(eff_val / 4.0, 1.0)
            stab = 1.0 if battle.active_pokemon and move.type in battle.active_pokemon.types else 0.0
            move_feats.extend([pow_val, eff_norm, stab])
        else:
            move_feats.extend([0.0, 0.0, 0.0])

    # 44-45: Faints
    my_faints_val = sum(1 for mon in battle.team.values() if mon.fainted) / 6.0
    opp_faints_val = sum(1 for mon in battle.opponent_team.values() if mon.fainted) / 6.0

    # 46-47: Status Conditions (Primary status)
    my_status = 1.0 if battle.active_pokemon and battle.active_pokemon.status else 0.0
    opp_status = 1.0 if battle.opponent_active_pokemon and battle.opponent_active_pokemon.status else 0.0

    obs = [turn_norm] + my_hp + opp_hp + my_boosts + opp_boosts + my_hazards + opp_hazards + [speed_adv] + move_feats + [my_faints_val, opp_faints_val, my_status, opp_status]
    
    arr = np.array(obs, dtype=np.float32)
    # Ensure exact size
    if arr.shape[0] < 48:
        arr = np.pad(arr, (0, 48 - arr.shape[0]))
    elif arr.shape[0] > 48:
        arr = arr[:48]
    
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
