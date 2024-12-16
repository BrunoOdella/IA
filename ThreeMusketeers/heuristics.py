from board import Board

def proximity_to_center(board: Board):
    """Evalúa la proximidad de los mosqueteros al centro del tablero."""
    center = (board.board_size[0] // 2, board.board_size[1] // 2)
    musketeer_positions = board.find_musketeer_positions()
    proximity_score = sum(
        -abs(pos[0] - center[0]) - abs(pos[1] - center[1])
        for pos in musketeer_positions
    )
    return proximity_score


def musketeer_mobility(board: Board):
    """Evalúa el número de movimientos disponibles para los mosqueteros."""
    return len(board.get_musketeer_valid_movements())


def enemy_mobility(board: Board):
    """Evalúa el número de movimientos disponibles para los enemigos."""
    return len(board.get_enemy_valid_movements())


def musketeers_alignment(board: Board):
    """Penaliza si los tres mosqueteros están alineados en fila o columna."""
    positions = board.find_musketeer_positions()
    if len(positions) != 3:
        return 0 
    rows = [pos[0] for pos in positions]
    cols = [pos[1] for pos in positions]
    if len(set(rows)) == 1 or len(set(cols)) == 1:
        return -100  
    return 0


def musketeer_on_trap(board: Board):
    """Penaliza si un mosquetero está sobre una trampa."""
    trap_position = board.find_trap_position()
    if trap_position in board.find_musketeer_positions():
        return -100  
    return 0



def enemy_count(board: Board):
    """Devuelve la diferencia entre el número de enemigos y mosqueteros (cuantos menos enemigos, mejor)."""
    enemy_count = - len(board.find_enemy_positions())
    return  enemy_count



def penalty_heuristic(board: Board):
    """Heurística que penaliza solo estados de derrota."""
    return musketeers_alignment(board) + musketeer_on_trap(board)


def penalty_heuristic_refined(board: Board):
    """
    Penaliza fuertemente estados terminales perdidos y estados previos que lleven a una derrota:
    Penaliza si los tres mosqueteros están alineados.
    Penaliza si un mosquetero está sobre una trampa.
    """
    is_done, winner = board.is_end(1) 
    if is_done:
        return -1000 if winner != 1 else 1000 


    return musketeers_alignment(board) + musketeer_on_trap(board)
