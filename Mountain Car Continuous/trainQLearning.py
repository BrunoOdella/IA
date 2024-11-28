import gymnasium as gym
import numpy as np
from qlearning_agent import QLearningAgent
from discretization import MountainCarDiscretizer, DiscretizationConfig
import matplotlib.pyplot as plt
from typing import List, Tuple
import pickle
from datetime import datetime


def train_agent(
    n_episodes: int = 1000, max_steps: int = 999, render: bool = False
) -> Tuple[QLearningAgent, List[float], List[float]]:
    """
    Entrena el agente Q-Learning en el ambiente Mountain Car Continuous

    Args:
        n_episodes: Número de episodios de entrenamiento
        max_steps: Máximo número de pasos por episodio
        render: Si se debe renderizar el ambiente

    Returns:
        Tuple con el agente entrenado y las listas de rewards y pasos por episodio
    """
    # Inicializar ambiente y agente
    env = gym.make("MountainCarContinuous-v0", render_mode="human" if render else None)
    discretizer = MountainCarDiscretizer()
    agent = QLearningAgent(
        discretizer=discretizer,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
    )

    # Métricas de entrenamiento
    episode_rewards = []
    episode_steps = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.learn(state, action, reward, next_state, done)

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        # Logging cada 10 episodios
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_steps = np.mean(episode_steps[-10:])
            print(f"Episodio {episode}")
            print(f"Reward promedio: {avg_reward:.2f}")
            print(f"Pasos promedio: {avg_steps:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}\n")

    env.close()
    return agent, episode_rewards, episode_steps


def plot_training_results(rewards: List[float], steps: List[float], window: int = 50):
    """
    Grafica los resultados del entrenamiento

    Args:
        rewards: Lista de rewards por episodio
        steps: Lista de pasos por episodio
        window: Tamaño de la ventana para el promedio móvil
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Calcular promedios móviles
    rolling_rewards = np.convolve(rewards, np.ones(window) / window, mode="valid")
    rolling_steps = np.convolve(steps, np.ones(window) / window, mode="valid")

    # Graficar rewards
    ax1.plot(rewards, alpha=0.3, label="Raw")
    ax1.plot(rolling_rewards, label=f"Media móvil ({window} episodios)")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Reward Total")
    ax1.legend()
    ax1.grid(True)

    # Graficar pasos
    ax2.plot(steps, alpha=0.3, label="Raw")
    ax2.plot(rolling_steps, label=f"Media móvil ({window} episodios)")
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Pasos")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.show()


def save_agent(agent: QLearningAgent, filename: str):
    """Guarda el agente entrenado"""
    with open(filename, "wb") as f:
        pickle.dump(agent, f)


if __name__ == "__main__":
    # Entrenar agente
    trained_agent, rewards, steps = train_agent(n_episodes=500)
    
    # Graficar resultados
    plot_training_results(rewards, steps)
    
    # Guardar agente
    save_agent(trained_agent, f'qlearning_agent_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')