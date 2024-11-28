import random
from collections import deque
from typing import Tuple, List
import numpy as np

class ExperienceBuffer:
    """
    Buffer circular para almacenar y muestrear experiencias de aprendizaje.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Args:
            max_size: Tamaño máximo del buffer
        """
        self.buffer = deque(maxlen=max_size)
        
    def add(self, experience: Tuple[np.ndarray, float, float, np.ndarray, bool]) -> None:
        """
        Añade una nueva experiencia al buffer.
        
        Args:
            experience: Tupla (estado, acción, recompensa, siguiente_estado, terminal)
        """
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Muestrea un batch de experiencias aleatorias.
        
        Args:
            batch_size: Tamaño del batch a muestrear
            
        Returns:
            Lista de experiencias muestreadas
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)