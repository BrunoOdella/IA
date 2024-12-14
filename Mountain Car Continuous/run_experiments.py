from experiment_manager import ExperimentManager, ExperimentConfig


def main():
    # Configuraciones de experimentos
    experiments = [
        # ExperimentConfig(
        #     learning_rate=0.1,
        #     discount_factor=0.95,
        #     epsilon=1.0,
        #     epsilon_decay=0.995,
        #     epsilon_min=0.01,
        #     n_episodes=500,
        #     max_steps=999,
        #     batch_size=32,  # Batch estándar para configuración baseline
        #     description="baseline",
        # ),
        # ExperimentConfig(
        #     learning_rate=0.05,
        #     discount_factor=0.99,
        #     epsilon=1.0,
        #     epsilon_decay=0.997,
        #     epsilon_min=0.01,
        #     n_episodes=500,
        #     max_steps=999,
        #     batch_size=64,  # Batch grande para aprendizaje conservador
        #     description="conservative_learning",
        # ),
        # ExperimentConfig(
        #     learning_rate=0.2,
        #     discount_factor=0.9,
        #     epsilon=1.0,
        #     epsilon_decay=0.99,
        #     epsilon_min=0.05,
        #     n_episodes=500,
        #     max_steps=999,
        #     batch_size=16,  # Batch pequeño para aprendizaje agresivo
        #     description="aggressive_learning",
        # ),
        # ExperimentConfig(
        #     learning_rate=0.2,
        #     discount_factor=0.99,
        #     epsilon=1.0,
        #     epsilon_decay=0.998,
        #     epsilon_min=0.05,
        #     n_episodes=1000,
        #     max_steps=500,
        #     batch_size=48,  # Batch intermedio para configuración optimizada
        #     description="optimized",
        # ),
        ExperimentConfig(
            learning_rate=0.03,
            discount_factor=0.98,
            epsilon=1.0,
            epsilon_decay=0.996,
            epsilon_min=0.01,
            n_episodes=800,
            max_steps=999,
            batch_size=64,  # Batch grande para aprendizaje estable
            description="slower_learning_high_discount | stochastic",
        ),
        ExperimentConfig(
            learning_rate=0.05,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.999,
            epsilon_min=0.01,
            n_episodes=800,
            max_steps=999,
            batch_size=32,  # Batch mediano para exploración extendida
            description="extended_exploration | stochastic",
        ),
        # ExperimentConfig(
        #     learning_rate=0.05,
        #     discount_factor=0.99,
        #     epsilon=1.0,
        #     epsilon_decay=0.997,
        #     epsilon_min=0.01,
        #     n_episodes=1000,
        #     max_steps=999,
        #     batch_size=48,  # Batch intermedio para entrenamiento largo
        #     description="longer_training",
        # ),
        # ExperimentConfig(
        #     learning_rate=0.05,
        #     discount_factor=0.99,
        #     epsilon=1.0,
        #     epsilon_decay=0.997,
        #     epsilon_min=0.05,
        #     n_episodes=500,
        #     max_steps=999,
        #     batch_size=32,  # Batch estándar para epsilon alto
        #     description="higher_min_epsilon",
        # ),
        ExperimentConfig(
            learning_rate=0.05,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.997,
            epsilon_min=0.01,
            n_episodes=1000,
            max_steps=500,
            batch_size=16,  # Batch pequeño para actualizaciones frecuentes
            description="more_episodes_fewer_steps | stochastic",
        ),
        ExperimentConfig(
            learning_rate=0.05,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.997,
            epsilon_min=0.01,
            n_episodes=800,
            max_steps=999,
            batch_size=48,  # Batch intermedio para aprendizaje adaptativo
            description="adaptive_learning_rate | stochastic",
        ),
    ]

    # Crear y ejecutar experimentos
    manager = ExperimentManager()

    for config in experiments:
        print(f"\nIniciando experimento: {config.description}")
        print("Configuración:")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Discount factor: {config.discount_factor}")
        print(f"  Epsilon decay: {config.epsilon_decay}")

        manager.run_experiment(config)
        manager.plot_experiment_results(config.description)

    # Guardar todos los resultados
    manager.save_results()
    print("\nExperimentos completados y resultados guardados!")


if __name__ == "__main__":
    main()