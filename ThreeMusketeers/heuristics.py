from board import Board

class Heuristics:
    @staticmethod
    def manhattan_heuristic(board: Board, player: int) -> float:
        musketeer_positions = board.find_musketeer_positions()
        trap_position = board.find_trap_position()

        proximity_penalty = sum(
            abs(musketeer_positions[i][0] - musketeer_positions[j][0]) +
            abs(musketeer_positions[i][1] - musketeer_positions[j][1])
            for i in range(len(musketeer_positions))
            for j in range(i + 1, len(musketeer_positions))
        )

        trap_penalty = sum(
            abs(pos[0] - trap_position[0]) + abs(pos[1] - trap_position[1])
            for pos in musketeer_positions
        ) if trap_position else 0

        return -proximity_penalty - trap_penalty

    @staticmethod
    def alignment_heuristic(board: Board, player: int) -> float:
        musketeer_positions = board.find_musketeer_positions()

        alignment_penalty = 0
        for i, pos1 in enumerate(musketeer_positions):
            for j, pos2 in enumerate(musketeer_positions):
                if i < j:
                    if pos1[0] == pos2[0] or pos1[1] == pos2[1]:  # Penaliza alineaciones
                        alignment_penalty += 1

        return -alignment_penalty

    @staticmethod
    def movement_heuristic(board: Board, player: int) -> float:
        musketeer_moves = len(board.get_musketeer_valid_movements())
        enemy_moves = len(board.get_enemy_valid_movements())
        return musketeer_moves - enemy_moves

    @staticmethod
    def center_proximity_heuristic(board: Board, player: int) -> float:
        musketeer_positions = board.find_musketeer_positions()
        center = (board.board_size[0] // 2, board.board_size[1] // 2)

        center_proximity = sum(
            abs(pos[0] - center[0]) + abs(pos[1] - center[1])
            for pos in musketeer_positions
        )

        return -center_proximity

    @staticmethod
    def zone_control_heuristic(board: Board, player: int) -> float:
        musketeer_positions = board.find_musketeer_positions()
        enemy_positions = board.find_enemy_positions()

        musketeer_control = len(musketeer_positions)
        enemy_control = len(enemy_positions)

        return musketeer_control - enemy_control
