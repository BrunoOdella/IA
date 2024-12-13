import random
import gymnasium as gym
import numpy as np
from board import Board

player_1 = 1
player_2 = 2

class ThreeMusketeersEnv(gym.Env):
    def __init__(self, grid_size=5, render_mode='rgb_array'):
        super(ThreeMusketeersEnv, self).__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.reset()
        
    def reset(self):
        self.current_player = player_1
        self.grid = Board((self.grid_size, self.grid_size))
        return self.grid

    def step(self, player, action):
        action_done = self.grid.play(player, action)
        
        if action_done:
            self.next_player()
        else:
            raise ValueError("Invalid action")
            
        done, winner = self.is_done()
        return self.grid, self.get_reward(), done, winner, self.grid.get_grid()
    
    def is_done(self):        
        return self.grid.is_end(self.current_player)
    
    def get_reward(self):
        is_done, winner = self.is_done()
        if is_done and winner == player_1:
            return 100
        elif is_done and winner == player_2:
            return -100
        
        return 0
        
    def next_player(self):
        self.current_player = player_1 if self.current_player == player_2 \
                                    else player_2
    def render(self):
        self.grid.render_grid(render_mode=self.render_mode)



