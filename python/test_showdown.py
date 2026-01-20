import asyncio
from poke_env.player import RandomPlayer


class LoggingBot(RandomPlayer):
    def choose_move(self, battle):
        obs = {
            "turn": battle.turn,
            "my_hp": [
                mon.current_hp_fraction if mon else 0.0
                for mon in battle.team.values()
            ],
            "opp_hp": [
                mon.current_hp_fraction if mon else 0.0
                for mon in battle.opponent_team.values()
            ],
            "available_moves": len(battle.available_moves),
            "available_switches": len(battle.available_switches),
        }

        print("OBS:", obs)
        return self.choose_random_move(battle)


async def main():
    bot = LoggingBot(
        battle_format="gen9randombattle",
        max_concurrent_battles=1,
    )

    opponent = RandomPlayer(
        battle_format="gen9randombattle",
        max_concurrent_battles=1,
    )

    print("Starting battle...")
    await bot.battle_against(opponent)
    print("Battle finished!")
    print("Bot battles:", bot.n_finished_battles)


if __name__ == "__main__":
    asyncio.run(main())
