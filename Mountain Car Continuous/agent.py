from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Tuple

class Agent(ABC):
    """Clase base abstracta para agentes de aprendizaje por refuerzo"""
    
    @abstractmethod
    def __init__(self):
        """Inicializa el agente"""
        pass
    
    @abstractmethod
    def get_action(self, state: np.ndarray) -> Any:
        pass
        
    @abstractmethod
    def learn(self, state: np.ndarray, action: Any, 
              reward: float, next_state: np.ndarray, 
              done: bool) -> None:
        pass