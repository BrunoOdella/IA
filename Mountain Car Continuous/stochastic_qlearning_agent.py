from typing import Tuple, Union
import numpy as np
from agent import Agent
from discretization import MountainCarDiscretizer
from experiencie_buffer import ExperienceBuffer

class StochasticQLearningAgent(Agent):
    def __init__(
        self,
        discretizer: MountainCarDiscretizer, # La discretización del espacio de estados y acciones
        learning_rate: float = 0.1, # es alfa de la formula
        discount_factor: float = 0.95, # es gamma de la formula
        epsilon: float = 1.0, # es para que el agente explore
        epsilon_decay: float = 0.995, # inicialmente el agente unicamente explora, pero con el tiempo explora menos
        epsilon_min: float = 0.01, # para que el agente no deje de explorar
        batch_size: int = 32, # tamaño del batch para el entrenamiento
        sample_size: int = None  # Tamaño del subconjunto para stoch_max
    ):
        super().__init__()
        self.discretizer = discretizer
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Definir el tamaño de muestreo como log(n) donde n es el número de acciones
        # sin el max se "alinea" al paper
        self.sample_size = sample_size  or int(np.log(discretizer.n_actions)) # or max(2, int(np.log(discretizer.n_actions))) Al menos una acción para exploración actual Al menos una acción para comparación/referencia

        # Memoria de acciones previas por estado | enfoque Memory-based | B.2.1. PROOF OF COROLLARY 5
        # combinar exploracion aleatoria con experiencia previa
        self.action_memory = {}
        
        self.q_table = np.zeros((
            discretizer.n_position_states,
            discretizer.n_velocity_states,
            discretizer.n_actions
        ))
        
        self.experience_buffer = ExperienceBuffer()
        
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
        
    def stoch_max(self, state_tuple: Tuple[int, int], return_action: bool = False) -> Union[float, Tuple[float, int]]:
        """
        Implementa stoch_max seleccionando el máximo Q-value de un subconjunto aleatorio
        de acciones más algunas acciones previamente utilizadas.
        
        Args:
            state_tuple: Tupla que representa el estado discretizado
            return_action: Si True, retorna también el índice de la acción
            
        Returns:
            Si return_action es False: máximo Q-value del subconjunto
            Si return_action es True: tupla (máximo Q-value, índice de acción)
        """
        
        # Obtener acciones aleatorias
        random_actions = np.random.choice(
            self.discretizer.n_actions, 
            size=self.sample_size, 
            replace=False
        )
        
        # Obtener acciones previas del estado (si existen)
        prev_actions = self.action_memory.get(state_tuple, set())
        action_set = set(random_actions) | prev_actions
        
        # Calcular Q-values y encontrar el máximo
        q_values = {}
    
        # Iteramos sobre cada acción posible en el conjunto de acciones
        for action in action_set:
            state_action_index = state_tuple + (action,)
            
            q_value = self.q_table[state_action_index]
            
            q_values[action] = q_value
        
        if not q_values:
            return (0.0, 0) if return_action else 0.0
        
        best_action = max(q_values.items(), key=lambda x: x[1])
        
        return (best_action[1], best_action[0]) if return_action else best_action[1]
    
    def get_action(self, state: np.ndarray) -> float:
        discrete_state = self.discretizer.discretize_state(state)
        
        if np.random.random() < self.epsilon:
            # Exploración: seleccionar acción aleatoria
            action_idx = np.random.randint(0, self.discretizer.n_actions)
        else:
            # Explotación: usar stoch_max para selección
            _, action_idx = self.stoch_max(discrete_state, return_action=True)
        
        # Actualizar memoria de acciones
        if discrete_state not in self.action_memory:
            self.action_memory[discrete_state] = set()
        self.action_memory[discrete_state].add(action_idx)
        if len(self.action_memory[discrete_state]) > self.sample_size:
            self.action_memory[discrete_state].pop()
            
        return self.discretizer.get_continuous_action(action_idx)
    
    def learn(self, state: np.ndarray, action: float, 
            reward: float, next_state: np.ndarray, 
            done: bool) -> None:
        # Almacenar experiencia
        self.experience_buffer.add((state, action, reward, next_state, done))
        
        # Solo aprender si tenemos suficientes experiencias
        if len(self.experience_buffer) >= self.batch_size:
            self._learn_from_batch()
            
    def _learn_from_batch(self) -> None:
        """Aprende de un batch de experiencias usando stoch_max"""
        experiences = self.experience_buffer.sample(self.batch_size)
        
        for state, action, reward, next_state, done in experiences:
            # Discretizar estados y acción
            curr_state = self.discretizer.discretize_state(state)
            next_state_disc = self.discretizer.discretize_state(next_state)
            action_idx = self.discretizer.discretize_action(action)
            
            # Usar stoch_max para la actualización
            best_next_q = self.stoch_max(next_state_disc)
            current_q = self.q_table[curr_state + (action_idx,)]
            
            # Actualización Q-value con stoch_max
            new_q = current_q + self.lr * (
                reward + self.gamma * best_next_q * (not done) - current_q
            )
            
            self.q_table[curr_state + (action_idx,)] = new_q
        
        # Actualizar epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay