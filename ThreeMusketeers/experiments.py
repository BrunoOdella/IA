from three_musketeers_env import ThreeMusketeersEnv
from agents import MinimaxAgent, ExpectimaxAgent
from captain_pete import CaptainPete
from heuristics import combined_heuristic, proximity_to_center, musketeer_mobility, penalty_heuristic, penalty_heuristic_refined
from play import play_multiple_games, plot_results
import pandas as pd

def run_experiments():
    env = ThreeMusketeersEnv()
     # input de profundidades
    depths = [5]
    # input de heurísticas
    heuristics = [penalty_heuristic_refined]
    heuristic_names = ["Penalty Heuristic Refined"]
    num_games = 100

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

    # Convertir resultados a DataFrame
    results_df = pd.DataFrame(results)

    # Mostrar gráficas individuales
    show_individual_results(results_df, num_games)

    return results_df

def show_individual_results(results_df, num_games):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")

    for _, row in results_df.iterrows():
        plt.figure(figsize=(8, 5))

        data = {
            "Player": [row["Agent"], "Captain Pete"],
            "Wins": [row["Wins"], row["Losses"]]
        }

        df = pd.DataFrame(data)
        bar_plot = sns.barplot(x="Player", y="Wins", data=df, palette="pastel", edgecolor="black")

        for p, value in zip(bar_plot.patches, data['Wins']):
            bar_plot.annotate(f'{value}', 
                              (p.get_x() + p.get_width() / 2., p.get_height()), 
                              ha='center', va='bottom', fontsize=12, color='black', rotation=0)

        plt.title(f"Results: {row['Agent']} (Depth {row['Depth']}, Heuristic: {row['Heuristic']})\nEach bar represents {num_games} games")
        plt.ylabel("Number of Wins")
        plt.xlabel("Players")
        plt.tight_layout()
        plt.show()

def plot_results_summary(results_df, num_games):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=results_df,
        x="Depth",
        y="Wins",
        hue="Agent",
        ci=None
    )

    plt.title(f"Resultados por Profundidad y Agente (Cada barra representa {num_games} partidas)")
    plt.ylabel("Partidas Ganadas")
    plt.xlabel("Profundidad")
    plt.legend(title="Agente")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = run_experiments()
    results.to_csv("experiment_results.csv", index=False)
