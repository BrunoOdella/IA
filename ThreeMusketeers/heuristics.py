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
    def enhanced_zone_control_heuristic(board: Board, player: int) -> float:
        # Parte 1: Control del tablero
        musketeer_positions = board.find_musketeer_positions()
        enemy_positions = board.find_enemy_positions()
        trap_position = board.find_trap_position()

        musketeer_control = len(musketeer_positions)
        enemy_control = len(enemy_positions)

        control_score = musketeer_control - enemy_control  # La base de zone control

        # Parte 2: Penalización por proximidad a la trampa
        trap_penalty = 0
        if trap_position:
            trap_penalty = sum(
                10 / (abs(pos[0] - trap_position[0]) + abs(pos[1] - trap_position[1]) + 1)
                for pos in musketeer_positions
            )

        # Parte 3: Penalización por alineaciones
        alignment_penalty = 0
        for i, pos1 in enumerate(musketeer_positions):
            for j, pos2 in enumerate(musketeer_positions):
                if i < j:
                    if pos1[0] == pos2[0]:  # Misma fila
                        alignment_penalty += 20
                    if pos1[1] == pos2[1]:  # Misma columna
                        alignment_penalty += 20

        # Parte 4: Incentivo por movimientos disponibles
        movement_bonus = len(board.get_musketeer_valid_movements())

        # Resultado final ponderado
        return control_score - trap_penalty - alignment_penalty + movement_bonus

    @staticmethod
    def enhanced_zone_control_with_strong_trap_penalty(board: Board, player: int) -> float:
        # Parte 1: Control del tablero
        musketeer_positions = board.find_musketeer_positions()
        enemy_positions = board.find_enemy_positions()
        trap_position = board.find_trap_position()

        musketeer_control = len(musketeer_positions)
        enemy_control = len(enemy_positions)

        control_score = musketeer_control - enemy_control

        # Parte 2: Penalización fuerte por trampas
        trap_penalty = 0
        if trap_position:
            for pos in musketeer_positions:
                distance_to_trap = abs(pos[0] - trap_position[0]) + abs(pos[1] - trap_position[1])
                
                # Penalización extrema si un mosquetero está en la trampa
                if distance_to_trap == 0:
                    return -1e6  # Pérdida inmediata
                
                # Penalización progresiva para la cercanía
                trap_penalty += 50 / distance_to_trap

        # Parte 3: Penalización por alineaciones
        alignment_penalty = 0
        for i, pos1 in enumerate(musketeer_positions):
            for j, pos2 in enumerate(musketeer_positions):
                if i < j:
                    if pos1[0] == pos2[0]:  # Misma fila
                        alignment_penalty += 20
                    if pos1[1] == pos2[1]:  # Misma columna
                        alignment_penalty += 20

        # Parte 4: Incentivo por movimientos disponibles
        movement_bonus = len(board.get_musketeer_valid_movements())

        # Resultado final ponderado
        return control_score - trap_penalty - alignment_penalty + movement_bonus
