import numpy as np
from typing import Tuple, List, Union
from dataclasses import dataclass

@dataclass
class DiscretizationConfig:
    """Configuration parameters for the Mountain Car discretization."""
    position_min: float = -1.2
    position_max: float = 0.6
    velocity_min: float = -0.07
    velocity_max: float = 0.07
    action_min: float = -1.0
    action_max: float = 1.0
    UMBRAL_10_GRADOS = 0.33
    UMBRAL_15_GRADOS = 0.5
    
class MountainCarDiscretizer:
    
    def __init__(self, config: DiscretizationConfig = DiscretizationConfig()):
        self.config = config
        self._setup_discretization()
        
    def _setup_discretization(self):
        """Initialize discretization boundaries and mappings."""
        # Action space discretization (logarithmic scale)
        self.actions = np.array([
            -1.0, -0.64, -0.32, -0.16, -0.08, -0.04, 0.0,
            0.04, 0.08, 0.16, 0.32, 0.64, 1.0
        ])
        
        # Position sectors based on terrain gradient
        self.position_boundaries = self._compute_position_sectors()
        
        # Velocity sectors based on energy considerations
        self.velocity_boundaries = np.array([
            -0.07, -0.05, -0.03, -0.01, -0.002, 0.002,
            0.01, 0.03, 0.05, 0.07
        ])

    def _compute_position_sectors(self) -> np.ndarray:
        """
        Compute adaptive position sectors based on terrain gradient.
        Uses the derivative of h(x) = sin(3x) to determine sector sizes.
        """
        # Generate initial uniform positions
        x = np.linspace(self.config.position_min, self.config.position_max, 100)
        # Compute terrain gradient
        gradient = 3 * np.cos(3 * x)
        # Identify critical points (where gradient changes significantly)
        critical_points = []
        
        # Add boundary points
        critical_points.append(self.config.position_min)
        
        # Add points where gradient changes rapidly
        for i in range(1, len(gradient)-1):
            if abs(gradient[i] - gradient[i-1]) > self.config.UMBRAL_15_GRADOS:
                critical_points.append(x[i])
                
        critical_points.append(self.config.position_max)
        
        return np.array(sorted(set(critical_points)))

    def discretize_state(self, state: np.ndarray) -> Tuple[int, int]:
        """
        Convert continuous state to discrete state indices.
        
        Args:
            state: Continuous state vector [position, velocity]
            
        Returns:
            Tuple of discrete indices (position_idx, velocity_idx)
        """
        position, velocity = state
        
        # Discretize position
        position_idx = np.digitize(position, self.position_boundaries) - 1
        
        # Discretize velocity
        velocity_idx = np.digitize(velocity, self.velocity_boundaries) - 1
        
        return position_idx, velocity_idx

    def discretize_action(self, action: Union[float, np.ndarray]) -> int:
        """
        Convert continuous action to discrete action index.
        
        Args:
            action: Continuous action value in [-1, 1]
            
        Returns:
            Index of closest discrete action
        """
        return np.abs(self.actions - action).argmin()

    def get_continuous_action(self, action_idx: int) -> float:
        """
        Convert discrete action index to continuous action value.
        
        Args:
            action_idx: Index in the discrete action space
            
        Returns:
            Corresponding continuous action value
        """
        return self.actions[action_idx]

    @property
    def n_position_states(self) -> int:
        """Number of discrete position states."""
        return len(self.position_boundaries) + 1

    @property
    def n_velocity_states(self) -> int:
        """Number of discrete velocity states."""
        return len(self.velocity_boundaries) + 1

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return len(self.actions)