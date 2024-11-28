import numpy as np
from typing import Tuple, Dict
from agent import Agent
from discretization import MountainCarDiscretizer

class QLearningAgent(Agent):
    def __init__(
        self,
        discretizer: MountainCarDiscretizer, # La discretización del espacio de estados y acciones
        learning_rate: float = 0.1, # es alfa de la formula
        discount_factor: float = 0.95, # es gamma de la formula
        epsilon: float = 1.0, # es para que el agente explore
        epsilon_decay: float = 0.995, # inicialmente el agente unicamente explora, pero con el tiempo explora menos
        epsilon_min: float = 0.01 # para que el agente no deje de explorar
    ):
        super().__init__()
        self.discretizer = discretizer
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table = np.zeros((
            discretizer.n_position_states,
            discretizer.n_velocity_states,
            discretizer.n_actions
        ))
    
    def get_action(self, state: np.ndarray) -> float:
        if np.random.random() < self.epsilon:
            # Exploración: seleccionar acción aleatoria del conjunto discreto
            action_idx = np.random.randint(0, self.discretizer.n_actions)
        else:
            # Explotación: seleccionar mejor acción según Q-table
            discrete_state = self.discretizer.discretize_state(state)
            action_idx = np.argmax(self.q_table[discrete_state])
            
        # Convertir índice de acción a valor continuo
        return self.discretizer.get_continuous_action(action_idx)
    
    def learn(self, state: np.ndarray, action: float, 
              reward: float, next_state: np.ndarray, 
              done: bool) -> None:
        # Discretizar estados y acción
        curr_state = self.discretizer.discretize_state(state)
        next_state_disc = self.discretizer.discretize_state(next_state)
        action_idx = self.discretizer.discretize_action(action)
        
        # Calcular valor Q siguiente máximo
        best_next_action = np.max(self.q_table[next_state_disc])
        current_q = self.q_table[curr_state + (action_idx,)]
        
        # Actualizar Q-value
        new_q = current_q + self.lr * (
            reward + self.gamma * best_next_action * (not done) - current_q
        )
        
        self.q_table[curr_state + (action_idx,)] = new_q
        
        # Actualizar epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay