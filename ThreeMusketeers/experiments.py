from three_musketeers_env import ThreeMusketeersEnv
from agents import MinimaxAgent, ExpectimaxAgent
from captain_pete import CaptainPete
from heuristics import musketeer_mobility, penalty_heuristic_refined, enemy_count, musketeers_alignment, musketeer_on_trap, proximity_to_center, enemy_mobility, penalty_heuristic
from play import play_multiple_games
import seaborn as sns
import matplotlib.pyplot as plt


def run_experiments():
    env = ThreeMusketeersEnv()
    # input de profundidades
    depths = [3]
    # input de heurísticas
    heuristics = [penalty_heuristic_refined]
    heuristic_names = ["Penalty Refined"]
    # input de cantidad de partidas
    num_games = 10

    results = []

    for depth in depths:
        for heuristic, heuristic_name in zip(heuristics, heuristic_names):
            minimax_agent = MinimaxAgent(player=1, depth=depth, heuristic=heuristic)
            expectimax_agent = ExpectimaxAgent(player=1, depth=depth, heuristic=heuristic)

            print(f"Running Minimax Agent with Depth {depth} and Heuristic '{heuristic_name}' ({num_games} games)")
            mm_wins, cp_wins = play_multiple_games(env, minimax_agent, CaptainPete(player=2), num_games)

            results.append({
                "Agent": "Minimax",
                "Depth": depth,
                "Heuristic": heuristic_name,
                "Wins": mm_wins,
                "Losses": cp_wins
            })

            print(f"Running Expectimax Agent with Depth {depth} and Heuristic '{heuristic_name}' ({num_games} games)")
            ex_wins, cp_wins_ex = play_multiple_games(env, expectimax_agent, CaptainPete(player=2), num_games)

            results.append({
                "Agent": "Expectimax",
                "Depth": depth,
                "Heuristic": heuristic_name,
                "Wins": ex_wins,
                "Losses": cp_wins_ex
            })

    show_individual_results(results, num_games)

    return results


def show_individual_results(results, num_games):
    sns.set_theme(style="whitegrid")

    for result in results:
        plt.figure(figsize=(8, 5))

        data = {
            "Player": [result["Agent"], "Captain Pete"],
            "Wins": [result["Wins"], result["Losses"]]
        }

        bar_plot = sns.barplot(x="Player", y="Wins", data=data, palette="pastel", edgecolor="black")

        for p, value in zip(bar_plot.patches, data['Wins']):
            bar_plot.annotate(f'{value}',
                              (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center', va='bottom', fontsize=12, color='black', rotation=0)

        plt.title(f"Results: {result['Agent']} (Depth {result['Depth']}, Heuristic: {result['Heuristic']})\nEach bar represents {num_games} games")
        plt.ylabel("Number of Wins")
        plt.xlabel("Players")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run_experiments()
