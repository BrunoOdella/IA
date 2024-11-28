from three_musketeers_env import ThreeMusketeersEnv
from minimax_agent import MinimaxAgent
from expectimax_agent import ExpectimaxAgent
from captain_pete import CaptainPete
from heuristics import Heuristics
from play import plot_individual_duel_results, play_multiple_games


def run_experiments():
    env = ThreeMusketeersEnv(grid_size=5)

    # Cambiar este código por {Heuristics.alignment_heuristic, Heuristics.movement_heuristic, etc.} para probar otra heuristica
    minimax_heuristic = Heuristics.enhanced_zone_control_with_strong_trap_penalty
    expectimax_heuristic = Heuristics.enhanced_zone_control_with_strong_trap_penalty

    # Crear agentes con las heurísticas seleccionadas
    minimax_agent = MinimaxAgent(player=1, heuristic=minimax_heuristic)
    expectimax_agent = ExpectimaxAgent(player=1, heuristic=expectimax_heuristic)
    captain_pete = CaptainPete(player=2)

    for depth in [5]:  # Ajustar profundidades mayores
        minimax_agent.depth = depth
        expectimax_agent.depth = depth

        print(f"Running experiments with depth={depth}...")

        # Minimax vs Captain Pete
        print("Minimax vs Captain Pete...")
        player1_wins_minimax, player2_wins_minimax = play_multiple_games(env, minimax_agent, captain_pete, num_games=1000)
        plot_individual_duel_results({
            "Model": ["Minimax", "Captain Pete"],
            "Wins": [player1_wins_minimax, player2_wins_minimax]
        }, duel_title=f"Minimax vs Captain Pete at Depth={depth}")

        # Expectimax vs Captain Pete
        print("Expectimax vs Captain Pete...")
        player1_wins_expectimax, player2_wins_expectimax = play_multiple_games(env, expectimax_agent, captain_pete, num_games=1000)
        plot_individual_duel_results({
            "Model": ["Expectimax", "Captain Pete"],
            "Wins": [player1_wins_expectimax, player2_wins_expectimax]
        }, duel_title=f"Expectimax vs Captain Pete at Depth={depth}")

        print(f"Depth={depth} completed:\n"
              f"Minimax wins: {player1_wins_minimax} | Captain Pete wins: {player2_wins_minimax} (vs Minimax)\n"
              f"Expectimax wins: {player1_wins_expectimax} | Captain Pete wins: {player2_wins_expectimax} (vs Expectimax)")


if __name__ == "__main__":
    run_experiments()
