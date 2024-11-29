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
            learning_rate=0.03,
            discount_factor=0.98,
            epsilon=1.0,
            epsilon_decay=0.996,
            epsilon_min=0.01,
            n_episodes=600,
            max_steps=999,
            description="slower_learning_high_discount", # keep - todas las metricas estan estables
        ),
        ExperimentConfig(
            learning_rate=0.05,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.999,
            epsilon_min=0.01,
            n_episodes=500,
            max_steps=999,
            description="extended_exploration", # keep - el periodo estable es muy corto para tener conclusiones, agregar episodios
        ),
        ExperimentConfig(
            learning_rate=0.05,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.997,
            epsilon_min=0.01,
            n_episodes=1000,
            max_steps=999,
            description="longer_training", # out - las metricas estan estables pero no es la mejor
        ),
        ExperimentConfig(
            learning_rate=0.05,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.997,
            epsilon_min=0.05,
            n_episodes=500,
            max_steps=999,
            description="higher_min_epsilon", # out - las metricas llegaron a un minimo y luego subieron
        ),
        ExperimentConfig(
            learning_rate=0.05,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.997,
            epsilon_min=0.01,
            n_episodes=800,
            max_steps=500,
            description="more_episodes_fewer_steps", # keep - mas episodios | las metricas tienden a mejorar con mas episodios | experiment_results_more_episodes_fewer_steps_20241128_210130.png
        ),
        ExperimentConfig(
            learning_rate=0.05,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.997,
            epsilon_min=0.01,
            n_episodes=500,
            max_steps=999,
            description="adaptive_learning_rate", # keep - con mas episodios pareceria mejorarm, tiene un comportamiento de amortiguamiento o oscilacion amortiguada
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
