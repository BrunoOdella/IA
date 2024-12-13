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

    for _ in tqdm(range(num_games)):
        winner = play_vs_other_agent(env, agent1, agent2, render)
        if winner == 1:
            player1_wins += 1
        else:
            player2_wins += 1

    return player1_wins, player2_wins

def plot_results(player1_wins, player2_wins):
    data = {'Players': ['Musketeers', 'Captain Pete'], 'Wins': [player1_wins, player2_wins]}
    df = pd.DataFrame(data)

    sns.set_theme(style="whitegrid") 
    plt.figure(figsize=(8, 5)) 
    bar_plot = sns.barplot(x='Players', y='Wins', hue='Players', data=df, palette='pastel', edgecolor='black', legend=False)

    plt.xlabel('Players', fontsize=14)
    plt.ylabel('Number of Wins', fontsize=14)
    plt.title('Number of Wins by Each Player', fontsize=16)

    for p in bar_plot.patches:
        bar_plot.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='bottom', fontsize=12, color='black', rotation=0)

    plt.tight_layout()
    plt.show()