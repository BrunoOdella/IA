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
        #     description="optimized",
        # ),
        ExperimentConfig(
            learning_rate=0.03,  # Disminuir tasa de aprendizaje
            discount_factor=0.98,  # Mantener el factor alto
            epsilon=1.0,
            epsilon_decay=0.996,
            epsilon_min=0.01,
            n_episodes=600,  # Aumentar el número de episodios para más exploración
            max_steps=999,
            description="slower_learning_high_discount",
        ),
        ExperimentConfig(
            learning_rate=0.05,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.999,  # Exploración más prolongada
            epsilon_min=0.01,
            n_episodes=500,
            max_steps=999,
            description="extended_exploration",
        ),
        ExperimentConfig(
            learning_rate=0.05,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.997,
            epsilon_min=0.01,
            n_episodes=1000,  # Más episodios
            max_steps=999,
            description="longer_training",
        ),
        ExperimentConfig(
            learning_rate=0.05,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.997,
            epsilon_min=0.05,  # Más exploración en la fase final
            n_episodes=500,
            max_steps=999,
            description="higher_min_epsilon",
        ),
        ExperimentConfig(
            learning_rate=0.05,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.997,
            epsilon_min=0.01,
            n_episodes=800,
            max_steps=500,  # Menos pasos pero más episodios
            description="more_episodes_fewer_steps",
        ),
        ExperimentConfig(
            learning_rate=0.05,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.997,
            epsilon_min=0.01,
            n_episodes=500,
            max_steps=999,
            description="adaptive_learning_rate",
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
