import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass
import pickle
import gymnasium as gym
from qlearning_agent import QLearningAgent
from discretization import MountainCarDiscretizer

@dataclass
class ExperimentConfig:
    """Configuración para un experimento de Q-Learning"""
    learning_rate: float
    discount_factor: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    n_episodes: int
    max_steps: int
    description: str

class ExperimentManager:
    """Gestor de experimentos para Q-Learning en Mountain Car"""
    
    def __init__(self):
        self.discretizer = MountainCarDiscretizer()
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_experiment(self, config: ExperimentConfig) -> Dict:
        """Ejecuta un experimento con la configuración dada"""
        env = gym.make("MountainCarContinuous-v0")
        agent = QLearningAgent(
            discretizer=self.discretizer,
            learning_rate=config.learning_rate,
            discount_factor=config.discount_factor,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay,
            epsilon_min=config.epsilon_min
        )
        
        metrics = {
            'episode_rewards': [],
            'episode_steps': [],
            'success_rate': [],
            'energy_consumed': [],
            'time_to_goal': []
        }
        
        success_window = []
        for episode in range(config.n_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            energy = 0
            
            for step in range(config.max_steps):
                action = agent.get_action(state)
                action_array = np.array([action])
                next_state, reward, terminated, truncated, _ = env.step(action_array)
                done = terminated or truncated
                
                modified_reward = self._compute_physics_based_reward(state, next_state, action, reward)
                
                agent.learn(state, action, modified_reward, next_state, done)
                
                total_reward += modified_reward
                steps += 1
                energy += action**2  # Energía consumida
                state = next_state
                
                if done:
                    break
            
            # Actualizar métricas
            metrics['episode_rewards'].append(total_reward)
            metrics['episode_steps'].append(steps)
            metrics['energy_consumed'].append(energy)
            
            # Éxito = llegó a la meta
            success = terminated and next_state[0] >= 0.45
            success_window.append(success)
            if len(success_window) > 100:
                success_window.pop(0)
            metrics['success_rate'].append(sum(success_window) / len(success_window))
            
            # Tiempo hasta la meta (si llegó)
            metrics['time_to_goal'].append(steps if success else config.max_steps)
            
            # Log cada 50 episodios
            if episode % 50 == 0:
                print(f"\nExperimento: {config.description}")
                print(f"Episodio: {episode}")
                print(f"Reward promedio últimos 50: {np.mean(metrics['episode_rewards'][-50:]):.2f}")
                print(f"Tasa de éxito: {metrics['success_rate'][-1]:.2%}")
        
        env.close()
        
        # Guardar resultados
        self.results[config.description] = {
            'config': config,
            'metrics': metrics,
            'agent': agent
        }
        
        return metrics
    
    def plot_experiment_results(self, description: str, window: int = 50):
        """Genera visualizaciones para un experimento específico"""
        if description not in self.results:
            raise ValueError(f"No se encontró el experimento {description}")
            
        metrics = self.results[description]['metrics']
        
        # Crear figura con subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Resultados del Experimento: {description}')
        
        # 1. Rewards
        self._plot_metric(axes[0,0], metrics['episode_rewards'], 
                         'Episodio', 'Reward Total', 'Rewards por Episodio', window)
        
        # 2. Pasos
        self._plot_metric(axes[0,1], metrics['episode_steps'],
                         'Episodio', 'Pasos', 'Pasos por Episodio', window)
        
        # 3. Tasa de éxito
        self._plot_metric(axes[1,0], metrics['success_rate'],
                         'Episodio', 'Tasa de Éxito', 'Tasa de Éxito por Episodio', window)
        
        # 4. Energía consumida
        self._plot_metric(axes[1,1], metrics['energy_consumed'],
                         'Episodio', 'Energía', 'Energía Consumida por Episodio', window)
        
        # 5. Tiempo hasta meta
        self._plot_metric(axes[2,0], metrics['time_to_goal'],
                         'Episodio', 'Pasos hasta Meta', 'Tiempo hasta Meta', window)
        
        # 6. Q-table heatmap
        agent = self.results[description]['agent']
        self._plot_qtable_heatmap(axes[2,1], agent.q_table)
        
        plt.tight_layout()
        plt.savefig(f'Mountain Car Continuous/Resultados/experiment_results_{description}_{self.timestamp}.png')
        plt.close()
        
    def _compute_physics_based_reward(self, 
                                    state: np.ndarray, 
                                    next_state: np.ndarray, 
                                    action: float, 
                                    base_reward: float) -> float:
        """
        Calcula una recompensa modificada basada en principios físicos del sistema.
        
        Args:
            state: Estado actual [posición, velocidad]
            next_state: Estado siguiente [posición, velocidad]
            action: Acción tomada
            base_reward: Recompensa base del ambiente
            
        Returns:
            float: Recompensa modificada
        """
        position, velocity = state
        next_position, next_velocity = next_state
        
        # 1. Recompensa por ganancia de energía potencial
        height_delta = next_position - position
        potential_energy_reward = height_delta * 15.0  # Factor de escala
        
        # 2. Recompensa por manejo eficiente de energía cinética
        velocity_reward = 0.0
        if position < -0.5:  # Zona de impulso inicial
            # Recompensar la acumulación de velocidad
            velocity_reward = abs(next_velocity) * 8.0
        elif -0.5 <= position < 0:  # Zona de acumulación de momento
            # Recompensar el movimiento oscilatorio
            velocity_reward = velocity * next_velocity * 12.0 if velocity * next_velocity < 0 else 0
        else:  # Zona de ascenso
            # Recompensar velocidad positiva
            velocity_reward = next_velocity * 10.0 if next_velocity > 0 else 0
            
        # 3. Penalización por uso excesivo de energía
        energy_penalty = -0.1 * (action**2)
        
        # 4. Bonus por alcanzar hitos
        milestone_bonus = 0.0
        if next_position >= 0.45:  # Meta alcanzada
            milestone_bonus = 100.0
        elif next_position >= 0.3 and velocity > 0:  # Progreso significativo
            milestone_bonus = 20.0
            
        return base_reward + potential_energy_reward + velocity_reward + energy_penalty + milestone_bonus 
        
    def _plot_metric(self, ax, data, xlabel, ylabel, title, window):
        """Utilidad para graficar una métrica con su media móvil"""
        ax.plot(data, alpha=0.3, label='Raw')
        rolling_mean = pd.Series(data).rolling(window=window).mean()
        ax.plot(rolling_mean, label=f'Media Móvil ({window})')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
    def _plot_qtable_heatmap(self, ax, q_table):
        """Genera un heatmap de la Q-table promediada sobre las acciones"""
        avg_q = np.mean(q_table, axis=2)
        sns.heatmap(avg_q.T, ax=ax, cmap='viridis')
        ax.set_title('Q-table Promedio por Estado')
        ax.set_xlabel('Índice Posición')
        ax.set_ylabel('Índice Velocidad')
        
    def save_results(self):
        """Guarda los resultados de todos los experimentos"""
        with open(f'Mountain Car Continuous/Resultados/experiment_results_{self.timestamp}.pkl', 'wb') as f:
            pickle.dump(self.results, f)