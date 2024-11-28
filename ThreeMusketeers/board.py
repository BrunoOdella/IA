import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

empty_cell = 0
musketeer = 1
enemy = 2
trap = 3

class Board:
    def __init__(self, board_size=(5, 5)):
        self.board_size = board_size
        self.create_board()
        self.first_movement = True
        
    def create_board(self):
        self.grid = np.full(self.board_size, enemy)
        self.place_musketeers()
        self.place_trap()

    def place_trap(self):
        enemy_cells = [(i, j) for i in range(self.board_size[0]) for j in range(self.board_size[1])
                       if self.grid[i][j] == enemy]
        chosen_cells = random.sample(enemy_cells, 1)

        for cell in chosen_cells:
            self.grid[cell[0]][cell[1]] = trap 
                
    def place_musketeers(self):
        n = self.board_size[0]
        musketeers = []
        
        while len(musketeers) < 3:
            row = random.randint(0, n-1)
            col = random.randint(0, n-1)
            
            if (row, col) in musketeers:
                continue 

            if len(musketeers) == 2:
                r1, c1 = musketeers[0]
                r2, c2 = musketeers[1]
                
                if (row == r1 == r2) or (col == c1 == c2):
                    continue 
            musketeers.append((row, col))
        
        for row, col in musketeers:
            self.grid[row][col] = musketeer
                    
    def get_grid(self):
        return tuple(self.grid.flatten().tolist())
    
    def grid_is_full(self):
        return not np.any(self.grid == empty_cell)

    def play(self, player, action):
        """Realiza una jugada en el tablero"""
        row = action[0]
        col = action[1]
        
        if row < 0 or row >= self.board_size[0] or col < 0 or col >= self.board_size[1]:
            return False
        
        if player == musketeer:
            valid_movements = self.get_musketeer_valid_movements()
            for origin_x, origin_y, x, y in valid_movements:
                if (x,y) == (row, col):
                    self.grid[row, col] = musketeer
                    self.grid[origin_x, origin_y] = empty_cell
                    return True
            
        else:
            valid_movements = self.get_enemy_valid_movements()
            for origin_x, origin_y, x, y in valid_movements:
                if (x,y) == (row, col):
                    self.grid[row, col] = enemy
                    self.grid[origin_x, origin_y] = empty_cell
                    return True
        
        return False
    
    def find_musketeer_positions(self):
        """Devuelve las posiciones de los mosqueteros"""
        musketeer_positions = []
        for row in range(self.board_size[0]):
            for col in range(self.board_size[1]):
                if self.grid[row, col] == musketeer:
                    musketeer_positions.append((row, col))
        return musketeer_positions
    
    def get_musketeer_valid_movements(self):
        """Devuelve los movimientos válidos para los mosqueteros"""
        musketeer_positions = self.find_musketeer_positions()
        valid_movements = []
        num_rows, num_cols = self.grid.shape
        
        for pos_x, pos_y in musketeer_positions:
            if pos_y + 1 < num_cols and (self.grid[pos_x, pos_y + 1] == enemy or self.grid[pos_x, pos_y + 1] == trap):
                valid_movements.append((pos_x, pos_y, pos_x, pos_y + 1))
            if pos_y - 1 >= 0 and (self.grid[pos_x, pos_y - 1] == enemy or self.grid[pos_x, pos_y - 1] == trap):
                valid_movements.append((pos_x, pos_y, pos_x, pos_y - 1))
            if pos_x + 1 < num_rows and (self.grid[pos_x + 1, pos_y] == enemy or self.grid[pos_x + 1, pos_y] == trap):
                valid_movements.append((pos_x, pos_y, pos_x + 1, pos_y))
            if pos_x - 1 >= 0 and (self.grid[pos_x - 1, pos_y] == enemy or self.grid[pos_x - 1, pos_y] == trap):
                valid_movements.append((pos_x, pos_y, pos_x - 1, pos_y))
        
        return valid_movements
    
    def find_enemy_positions(self):
        """Devuelve las posiciones de los enemigos"""
        enemy_positions = []
        for row in range(self.board_size[0]):
            for col in range(self.board_size[1]):
                if self.grid[row, col] == enemy:
                    enemy_positions.append((row, col))
                    
        return enemy_positions
    
    
    def find_trap_position(self):
        """Devuelve la posición de la trampa"""
        for row in range(self.board_size[0]):
            for col in range(self.board_size[1]):
                if self.grid[row, col] == trap:
                    return (row, col)
                    
        return None
                
    def get_enemy_valid_movements(self):
        """Devuelve los movimientos válidos para los enemigos"""
        enemy_positions = self.find_enemy_positions()
        valid_movements = []
        num_rows, num_cols = self.grid.shape
        
        for pos_x, pos_y in enemy_positions:
            if pos_y + 1 < num_cols and self.grid[pos_x, pos_y + 1] == empty_cell:
                valid_movements.append((pos_x, pos_y, pos_x, pos_y + 1))
            if pos_y - 1 >= 0 and self.grid[pos_x, pos_y - 1] == empty_cell:
                valid_movements.append((pos_x, pos_y, pos_x, pos_y - 1))
            if pos_x + 1 < num_rows and self.grid[pos_x + 1, pos_y] == empty_cell:
                valid_movements.append((pos_x, pos_y, pos_x + 1, pos_y))
            if pos_x - 1 >= 0 and self.grid[pos_x - 1, pos_y] == empty_cell:
                valid_movements.append((pos_x, pos_y, pos_x - 1, pos_y))
        
        return valid_movements
    
    def render_grid(self, render_mode):
        """Imprime el tablero"""
        rgb_array = self.render_rgb_array()
        plt.imshow(rgb_array)
        plt.show()
        
    def render_rgb_array(self):
        """Genera un array RGB de la representación del tablero."""
        rgb_array = np.zeros((self.board_size[0] * 10, self.board_size[1] * 10, 3), dtype=np.uint8)

        background_color = [255, 99, 71]  

        musketeer_color = [0, 0, 0] 
        enemy_color = [255, 255, 255]
        trap_color = [255, 0, 0] 

        rgb_array[:] = background_color

        for row in range(self.board_size[0]):
            for col in range(self.board_size[1]):
                center_x = col * 10 + 5
                center_y = row * 10 + 5

                if self.grid[row, col] == musketeer:
                    cv2.circle(rgb_array, (center_x, center_y), 5, musketeer_color, -1)
                elif self.grid[row, col] == enemy:
                    cv2.circle(rgb_array, (center_x, center_y), 5, enemy_color, -1)
                elif self.grid[row, col] == trap:
                    cv2.circle(rgb_array, (center_x, center_y), 5, trap_color, -1)
                

        return rgb_array
    
    def clone(self):
        """Devuelve una copia del tablero"""
        board_clone = Board(board_size=self.board_size)
        board_clone.grid = np.copy(self.grid)
        return board_clone
    
    def is_end(self, player):
        """
        Devuelve: (True/False que indica si termino el juego, 1/2 que indica el jugador que ganó)
        """
        # Check if musketeers have no movements left
        musketeer_movements = self.get_musketeer_valid_movements()
        if len(musketeer_movements) == 0:
            return True, musketeer
        
        # Check if are enemies left
        enemies = self.find_enemy_positions()
        if len(enemies) == 0:
            return True, musketeer
           
        # Check if musketeers are aligned in either a row or a column
        musketeer_positions = self.find_musketeer_positions()
        rows = [pos[0] for pos in musketeer_positions]  
        cols = [pos[1] for pos in musketeer_positions]  

        if len(set(rows)) == 1 or len(set(cols)) == 1: 
            return True, enemy
        
        # Check if enemies have no movements left
        enemy_movements = self.get_enemy_valid_movements()
        if len(enemy_movements) == 0 and not(self.grid_is_full()):
            return True, musketeer

        # Check if one musketeer is in the trap
        trap_pos = self.find_trap_position()  
        if trap_pos is None:
            return True, enemy 
          
        return False, 0
    
    def get_possible_actions(self, player):
        """Devuelve una lista de acciones posibles para el jugador"""
        if player == musketeer:
            possible_movements = self.get_musketeer_valid_movements()
            possible_actions = []
            for origin_x, origin_y, x, y in possible_movements:
                possible_actions.append((x, y))
        else:
            possible_movements = self.get_enemy_valid_movements()
            possible_actions = []
            for origin_x, origin_y, x, y in possible_movements:
                possible_actions.append((x, y))
            
            
        return possible_actions