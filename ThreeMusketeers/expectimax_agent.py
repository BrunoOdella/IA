from agent import Agent
from board import Board
from typing import Callable

class ExpectimaxAgent(Agent):
    def __init__(self, player, heuristic: Callable[[Board, int], float]):
        super().__init__(player)
        self.heuristic = heuristic

    def next_action(self, board: Board):
        _, action = self.expectimax(board, depth=3, maximizing_player=True)
        return action

    def expectimax(self, board: Board, depth, maximizing_player):
        if depth == 0 or board.is_end(self.player)[0]:
            return self.heuristic_utility(board), None

        if maximizing_player:
            max_value = float('-inf')
            best_action = None
            for action in board.get_possible_actions(self.player):
                new_board = board.clone()
                new_board.play(self.player, action)
                value, _ = self.expectimax(new_board, depth - 1, False)
                if value > max_value:
                    max_value = value
                    best_action = action
            return max_value, best_action
        else:
            expected_value = 0
            enemy = (self.player % 2) + 1
            possible_actions = board.get_possible_actions(enemy)
            for action in possible_actions:
                new_board = board.clone()
                new_board.play(enemy, action)
                value, _ = self.expectimax(new_board, depth - 1, True)
                expected_value += value / len(possible_actions)
            return expected_value, None

    def heuristic_utility(self, board: Board):
        # Llama a la heurística definida en la inicialización
        return self.heuristic(board, self.player)
