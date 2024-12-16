from agent import Agent
import math

class MinimaxAgent(Agent):
    def __init__(self, player, depth, heuristic):
        super().__init__(player)
        self.depth = depth
        self.heuristic = heuristic

    def next_action(self, obs):
        _, action = self.minimax(obs, self.depth, -math.inf, math.inf, True)
        return action

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_end(self.player)[0]:
            return self.heuristic(board), None

        if maximizing_player:
            max_eval = -math.inf
            best_action = None
            for action in board.get_possible_actions(self.player):
                next_board = board.clone()
                next_board.play(self.player, action)
                eval, _ = self.minimax(next_board, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else:
            min_eval = math.inf
            best_action = None
            opponent = (self.player % 2) + 1
            for action in board.get_possible_actions(opponent):
                next_board = board.clone()
                next_board.play(opponent, action)
                eval, _ = self.minimax(next_board, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_action

    def heuristic_utility(self, board):
        return self.heuristic(board)

class ExpectimaxAgent(Agent):
    def __init__(self, player, depth, heuristic):
        super().__init__(player)
        self.depth = depth
        self.heuristic = heuristic

    def next_action(self, obs):
        _, action = self.expectimax(obs, self.depth, True)
        return action

    def expectimax(self, board, depth, maximizing_player):
        if depth == 0 or board.is_end(self.player)[0]:
            return self.heuristic(board), None

        if maximizing_player:
            max_eval = -math.inf
            best_action = None
            for action in board.get_possible_actions(self.player):
                next_board = board.clone()
                next_board.play(self.player, action)
                eval, _ = self.expectimax(next_board, depth - 1, False)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
            return max_eval, best_action
        else:
            total_eval = 0
            actions = board.get_possible_actions((self.player % 2) + 1)
            if not actions:
                return self.heuristic(board), None
            probability = 1 / len(actions)
            for action in actions:
                next_board = board.clone()
                next_board.play((self.player % 2) + 1, action)
                eval, _ = self.expectimax(next_board, depth - 1, True)
                total_eval += probability * eval
            return total_eval, None

    def heuristic_utility(self, board):
        return self.heuristic(board)
