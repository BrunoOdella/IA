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
        """
        Selecciona una acción basada en el estado actual
        
        Args:
            state: El estado actual del ambiente
            
        Returns:
            La acción seleccionada
        """
        pass
        
    @abstractmethod
    def learn(self, state: np.ndarray, action: Any, 
              reward: float, next_state: np.ndarray, 
              done: bool) -> None:
        """
        Actualiza el conocimiento del agente basado en la experiencia
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Estado siguiente
            done: Indicador de fin de episodio
        """
        pass