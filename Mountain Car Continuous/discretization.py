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
        self.actions = np.array(
            [
                -1.0,
                -0.8,
                -0.6,
                -0.4,
                -0.2,
                -0.1,
                -0.05,
                0.0,
                0.05,
                0.1,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ]
        )

        # Position sectors based on terrain gradient
        self.position_boundaries = self._compute_position_sectors()

        # Velocity sectors based on energy considerations
        self.velocity_boundaries = np.array(
            [
                -0.07,
                -0.05,
                -0.03,
                -0.02,
                -0.01,
                -0.005,
                0.0,
                0.005,
                0.01,
                0.02,
                0.03,
                0.05,
                0.07,
            ]
        )


    def _compute_position_sectors(self) -> np.ndarray:
        """
        Calcula sectores de posición adaptativos basados en:
        1. Gradiente del terreno (derivada de h(x) = sin(3x))
        2. Distancia a la meta (resolución más fina cerca de x = 0.6)
        """
        # Generar posiciones uniformes iniciales
        x = np.linspace(
            self.config.position_min, self.config.position_max, 200
        )  

        # Calcular gradiente del terreno
        gradient = 3 * np.cos(3 * x)
        critical_points = set() 

        # 1. Añadir puntos basados en el gradiente
        critical_points.add(self.config.position_min)
        for i in range(1, len(gradient) - 1):
            if abs(gradient[i] - gradient[i - 1]) > self.config.UMBRAL_15_GRADOS:
                critical_points.add(x[i])

        # 2. Añadir puntos con mayor resolución cerca de la meta
        goal_position = self.config.position_max
        for distance_to_goal in [0.4, 0.3, 0.2, 0.1, 0.05, 0.025]:
            critical_points.add(goal_position - distance_to_goal)

        # Añadir punto final
        critical_points.add(self.config.position_max)

        # Ordenar todos los puntos
        return np.array(sorted(critical_points))

    def discretize_state(self, state: np.ndarray) -> Tuple[int, int]:
        position, velocity = state

        # Discretize position
        position_idx = np.digitize(position, self.position_boundaries) - 1

        # Discretize velocity
        velocity_idx = np.digitize(velocity, self.velocity_boundaries) - 1

        return position_idx, velocity_idx

    def discretize_action(self, action: Union[float, np.ndarray]) -> int:
        return np.abs(self.actions - action).argmin()

    def get_continuous_action(self, action_idx: int) -> float:
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
