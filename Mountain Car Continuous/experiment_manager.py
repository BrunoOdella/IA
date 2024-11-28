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
                # Convertir la acción escalar a array numpy de forma (1,)
                action_array = np.array([action])
                next_state, reward, terminated, truncated, _ = env.step(action_array)
                done = terminated or truncated
                
                agent.learn(state, action, reward, next_state, done)
                
                total_reward += reward
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
        plt.savefig(f'experiment_results_{description}_{self.timestamp}.png')
        plt.close()
        
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
        with open(f'experiment_results_{self.timestamp}.pkl', 'wb') as f:
            pickle.dump(self.results, f)