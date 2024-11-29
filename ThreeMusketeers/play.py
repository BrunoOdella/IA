from datetime import datetime
from board import Board as GameBoard
from random_agent import RandomAgent
from agent import Agent
from three_musketeers_env import ThreeMusketeersEnv
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd


def play_vs_other_agent(env, agent1, agent2, render=False, verbose=False):
    done = False
    obs = env.reset()
    start = datetime.now()
    winner = 0
    player_1 = 1
    player_2 = 2
    while not done:
        if render: env.render()
        action = agent1.next_action(obs)
        obs, _, done, winner, _ = env.step(player_1, action)
        if render: env.render()
        if not done:
            next_action = agent2.next_action(obs)
            _, _, done, winner, _ = env.step(player_2, next_action)
    if render: env.render()
    if verbose: print('------ Total time: {}\n'.format(datetime.now() - start))
    if winner == 1:
        if verbose: print('------ Player 1 won')
    else:
        if verbose: print('------ Player 2 (opponent) won')
        
    return winner


def play_multiple_games(env, agent1, agent2, num_games=100, render=False):
    player1_wins = 0
    player2_wins = 0

    with tqdm(total=num_games, desc="Playing games") as pbar:
        for _ in range(num_games):
            winner = play_vs_other_agent(env, agent1, agent2, render)
            if winner == 1:
                player1_wins += 1
            else:
                player2_wins += 1

            # Actualizar la barra de progreso con victorias y derrotas
            pbar.set_postfix({
                "Player 1 Wins": player1_wins,
                "Player 2 Wins": player2_wins
            })
            pbar.update(1)

    return player1_wins, player2_wins

def plot_results(results):
    """
    Plot cumulative results comparing agents across multiple depths.
    """
    df = pd.DataFrame(results)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Crear gráfico de barras agrupado por agente y profundidad
    bar_plot = sns.barplot(
        data=df,
        x="Depth",
        y="Wins",
        hue="Player",
        palette="husl",
        edgecolor="black"
    )

    plt.title("Comparison of Player Performances Across Depths", fontsize=16)
    plt.xlabel("Search Depth", fontsize=14)
    plt.ylabel("Number of Wins", fontsize=14)

    # Anotar valores encima de las barras
    for p in bar_plot.patches:
        bar_plot.annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=10, color='black'
        )

    plt.legend(title="Player", loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_individual_duel_results(results, duel_title):
    """
    Genera un gráfico de resultados para un duelo individual entre dos modelos.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # Convertir resultados a DataFrame
    df = pd.DataFrame(results)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))  # Ajusta el tamaño del gráfico

    # Crear gráfico de barras
    bar_plot = sns.barplot(
        data=df,
        x="Model",
        y="Wins",
        palette="pastel",
        edgecolor="black"
    )

    # Títulos y etiquetas
    plt.title(duel_title, fontsize=16)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Number of Wins", fontsize=12)

    # Anotar valores encima de las barras
    for p in bar_plot.patches:
        bar_plot.annotate(
            f'{int(p.get_height())}',  # Número de victorias
            (p.get_x() + p.get_width() / 2., p.get_height()),  # Coordenadas
            ha='center', va='bottom', fontsize=10, color='black'
        )

    plt.tight_layout()
    plt.show()
