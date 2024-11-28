from agent import Agent
from board import Board
from typing import Callable

class MinimaxAgent(Agent):
    def __init__(self, player, heuristic: Callable[[Board, int], float]):
        super().__init__(player)
        self.heuristic = heuristic

    def next_action(self, board: Board):
        _, action = self.minimax(board, depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)
        return action

    def minimax(self, board: Board, depth, alpha, beta, maximizing_player):
        # Chequear fin del juego
        is_end, winner = board.is_end(self.player)
        if is_end:
            # Pérdida si el oponente gana
            return (-1e6 if winner != self.player else 1e6), None

        # Profundidad base
        if depth == 0:
            return self.heuristic_utility(board, depth), None

        # Maximizar o minimizar según el jugador
        if maximizing_player:
            max_value = float('-inf')
            best_action = None
            for action in board.get_possible_actions(self.player):
                new_board = board.clone()
                new_board.play(self.player, action)
                value, _ = self.minimax(new_board, depth - 1, alpha, beta, False)
                if value > max_value:
                    max_value = value
                    best_action = action
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return max_value, best_action
        else:
            min_value = float('inf')
            enemy = (self.player % 2) + 1
            for action in board.get_possible_actions(enemy):
                new_board = board.clone()
                new_board.play(enemy, action)
                value, _ = self.minimax(new_board, depth - 1, alpha, beta, True)
                if value < min_value:
                    min_value = value
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return min_value, None

    def heuristic_utility(self, board: Board, depth: int):
        return self.heuristic(board, self.player, depth)
