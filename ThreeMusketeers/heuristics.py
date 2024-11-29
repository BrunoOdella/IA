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

    @staticmethod
    def enhanced_zone_control_with_dynamic_movement_evaluation(board: Board, player: int) -> float:
        # Parte 1: Control del tablero
        musketeer_positions = board.find_musketeer_positions()
        enemy_positions = board.find_enemy_positions()
        trap_position = board.find_trap_position()

        musketeer_control = len(musketeer_positions)
        enemy_control = len(enemy_positions)

        control_score = musketeer_control - enemy_control

        # Parte 2: Penalización fuerte por trampas
        trap_penalty = 0
        safe_bonus = 0
        if trap_position:
            for pos in musketeer_positions:
                distance_to_trap = abs(pos[0] - trap_position[0]) + abs(pos[1] - trap_position[1])

                if distance_to_trap == 0:
                    return -1e6  # Pérdida inmediata por estar en la trampa

                # Penalización progresiva por cercanía
                trap_penalty += 100 / (distance_to_trap ** 2 + 1)

                # Incentivo por mantenerse alejado
                safe_bonus += 10 * distance_to_trap

            # Penalizar movimientos hacia la trampa
            for move in board.get_musketeer_valid_movements():
                _, _, x, y = move
                if (x, y) == trap_position:
                    trap_penalty += 1e4

        # Parte 3: Penalización por alineaciones
        alignment_penalty = 0
        for i, pos1 in enumerate(musketeer_positions):
            for j, pos2 in enumerate(musketeer_positions):
                if i < j:
                    if pos1[0] == pos2[0]:  # Misma fila
                        alignment_penalty += 50
                    if pos1[1] == pos2[1]:  # Misma columna
                        alignment_penalty += 50

        # Parte 4: Evaluación dinámica de movimientos
        musketeer_moves = len(board.get_musketeer_valid_movements())
        enemy_moves = len(board.get_enemy_valid_movements())
        movement_score = musketeer_moves - enemy_moves  # Diferencia de movimientos

        # Resultado final ponderado
        return control_score - trap_penalty - alignment_penalty + movement_score + safe_bonus
