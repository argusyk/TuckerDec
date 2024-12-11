import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ініціалізація матриць T для трьох критеріїв
T_1 = tf.constant([[0.9, 0.8, 0.7],
                   [0.85, 0.8, 0.75],
                   [0.7, 0.6, 0.5]], dtype=tf.float32)

T_2 = tf.constant([[0.7, 0.6, 0.5],
                   [0.6, 0.55, 0.5],
                   [0.4, 0.3, 0.2]], dtype=tf.float32)

T_3 = tf.constant([[0.8, 0.7, 0.6],
                   [0.75, 0.65, 0.55],
                   [0.5, 0.4, 0.3]], dtype=tf.float32)

# Ініціалізація векторів загроз Z для трьох критеріїв
Z_1 = tf.constant([0.9, 0.8, 0.7, 0.6], dtype=tf.float32)
Z_2 = tf.constant([0.85, 0.75, 0.65, 0.55], dtype=tf.float32)
Z_3 = tf.constant([0.8, 0.7, 0.6, 0.5], dtype=tf.float32)

# Функція для обчислення тензора S
def calculate_s_tensor(T, Z):
    """
    Розрахунок матриць S для заданого критерію T та вектора загроз Z.
    """
    S = []
    for z in Z:
        S.append(tf.round(T * z * 100) / 100)  # Округлення до двох знаків після крапки
    return tf.stack(S)  # Повертаємо 3D-тензор, де кожен шар відповідає одній загрозі

# Функція для моделювання катастрофічних змін у тензорі T
def modify_tensor(T, stage, factor):
    """
    Плавна модифікація матриці T відповідно до стадії катастрофи.
    """
    if stage == 'before':
        return T
    elif stage == 'during':
        return T * (1 - factor)
    elif stage == 'after':
        return T * (0.5 + 0.5 * factor)
    else:
        raise ValueError("Unknown stage. Use 'before', 'during', or 'after'.")

# Функція для обчислення та відображення мінімальних та максимальних значень тензора S
def plot_min_max(S, criterion_name, subsystems_labels, stages_labels):
    """
    Побудова графіків мінімуму та максимуму для тензора S.
    """
    num_stages = S.shape[0]
    num_subsystems = S.shape[2]

    min_vals = np.min(S.numpy(), axis=1)
    max_vals = np.max(S.numpy(), axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(num_subsystems):
        ax.plot(stages_labels, min_vals[:, i], label=f"{subsystems_labels[i]} (Min)", linestyle='--', marker='o')
        ax.plot(stages_labels, max_vals[:, i], label=f"{subsystems_labels[i]} (Max)", linestyle='-', marker='s')
    
    ax.set_title(f"Мінімум та Максимум для {criterion_name}")
    ax.set_xlabel("Стадія катастрофи")
    ax.set_ylabel("Значення S")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Основна функція для запуску експериментів
def run_disaster_experiment(T, Z, criterion_name):
    """
    Експеримент із моделювання сценаріїв катастроф для одного критерію.
    """
    stages = ['before', 'during', 'after']
    stage_factors = {'before': 0.0, 'during': 0.5, 'after': 0.8}
    subsystems_labels = ['Генерація', 'Транспортування', 'Споживання']
    stages_labels = ['До катастрофи', 'Під час катастрофи', 'Після катастрофи']

    S_total = []
    for stage in stages:
        modified_T = modify_tensor(T, stage, stage_factors[stage])
        S_stage = calculate_s_tensor(modified_T, Z)
        S_total.append(S_stage)

    S_total = tf.stack(S_total)  # [stages, threats, subsystems]
    plot_min_max(S_total, criterion_name, subsystems_labels, stages_labels)

# Функція для побудови графіків Min/Max - Threat Level
def plot_min_max_threat_level(S, Z, criterion_name):
    """
    Побудова графіка мінімальних і максимальних значень S залежно від рівнів загроз Z.
    
    Parameters:
        S: tf.Tensor
            Тензор розмірності [threat_levels, subsystems, ...].
        Z: tf.Tensor
            Вектор рівнів загроз.
        criterion_name: str
            Назва критерію для заголовку графіка.
    """
    num_threats = Z.shape[0]
    num_subsystems = S.shape[1]

    min_vals = tf.reduce_min(S, axis=2).numpy()  # Мінімальні значення по елементах кожної підсистеми
    max_vals = tf.reduce_max(S, axis=2).numpy()  # Максимальні значення по елементах кожної підсистеми

    subsystems_labels = ['Генерація', 'Транспортування', 'Споживання']

    plt.figure(figsize=(10, 6))
    for i in range(num_subsystems):
        plt.plot(Z.numpy(), min_vals[:, i], label=f'{subsystems_labels[i]} Мінімум', linestyle='--', marker='o')
        plt.plot(Z.numpy(), max_vals[:, i], label=f'{subsystems_labels[i]} Максимум', linestyle='-', marker='s')

    plt.xlabel('Рівень загрози (Threat Level)')
    plt.ylabel('Значення (Min / Max)')
    plt.title(f"Min/Max значення для {criterion_name} залежно від рівнів загроз")
    plt.legend()
    plt.grid(True)
    plt.show()

# Використання функції
def run_min_max_threat_experiment():
    """
    Виконання експерименту з побудовою графіків Min/Max - Threat Level.
    """
    criterion_names = ["Criterion 1 (Infrastructure Functionality)",
                       "Criterion 2 (Recovery Ability)",
                       "Criterion 3 (Resilience to Threats)"]
    Ts = [T_1, T_2, T_3]
    Zs = [Z_1, Z_2, Z_3]

    for i in range(len(Ts)):
        T = Ts[i]
        Z = Zs[i]
        criterion_name = criterion_names[i]

        # Розрахунок тензора S для кожного критерію
        S = calculate_s_tensor(T, Z)

        # Побудова графіків Min/Max - Threat Level
        plot_min_max_threat_level(S, Z, criterion_name)

# Функція для побудови 3D поверхонь для значень S залежно від рівня загроз
def plot_3d_surface_for_threat_level(S, Z, criterion_name):
    """
    Побудова 3D поверхні значень S залежно від рівня загроз Z для кожного критерію.
    
    Parameters:
        S: tf.Tensor
            Тензор розмірності [threat_levels, subsystems, elements_per_subsystem].
        Z: tf.Tensor
            Вектор рівнів загроз.
        criterion_name: str
            Назва критерію для заголовку графіка.
    """
    num_threats = Z.shape[0]
    num_subsystems = S.shape[1]
    num_elements = S.shape[2]

    # Створення сітки для 3D графіка
    subsystems_labels = ['Генерація', 'Транспортування', 'Споживання']
    elements_labels = ['Після катастрофи', 'Під час катастрофи', 'До катастрофи']

    X = np.arange(num_subsystems)  # Вісь підсистем (0, 1, 2)
    Y = np.arange(num_elements)   # Елементи в підсистемах
    X, Y = np.meshgrid(X, Y)

    # Створення фігури для 3D графіку
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Побудова 3D поверхні для кожного рівня загроз
    for threat_level in range(num_threats):
        Z_surface = S[threat_level].numpy()  # Отримуємо значення для поточного рівня загроз
        ax.plot_surface(X, Y, Z_surface, alpha=0.7, cmap='viridis', edgecolor='k')

    ax.set_xlabel("Підсистеми")
    ax.set_ylabel("Стадія катастрофи")
    ax.set_zlabel("Значення S")
    ax.set_title(f"3D поверхня для {criterion_name} на різних рівнях загроз")

    # Встановлення значень для осей X та Y
    ax.set_xticks(np.arange(num_subsystems))  # Тики на осі X (0, 1, 2)
    ax.set_xticklabels(subsystems_labels)  # Підпис осі X
    ax.set_yticks(np.arange(num_elements))  # Тики на осі Y (0, 1, 2)
    ax.set_yticklabels(elements_labels)  # Підпис осі Y

    plt.show()

# Функція для запуску експерименту з побудовою 3D поверхні для кожного критерію
def run_3d_surface_for_all_criteria():
    """
    Запуск експерименту з побудовою 3D поверхонь для кожного критерію.
    """
    criterion_names = ["Criterion 1 (Infrastructure Functionality)",
                       "Criterion 2 (Recovery Ability)",
                       "Criterion 3 (Resilience to Threats)"]
    Ts = [T_1, T_2, T_3]
    Zs = [Z_1, Z_2, Z_3]

    for i in range(len(Ts)):
        T = Ts[i]
        Z = Zs[i]
        criterion_name = criterion_names[i]

        # Розрахунок тензора S для кожного критерію
        S = calculate_s_tensor(T, Z)

        # Побудова 3D поверхні для поточного критерію
        plot_3d_surface_for_threat_level(S, Z, criterion_name)

# Функція для побудови теплової карти значень тензора S
def plot_heatmap(S, Z, stage, criterion_name, threat_level_labels):
    """
    Побудова теплової карти значень тензора S для заданої стадії катастрофи з відображенням значення Z.
    
    Parameters:
        S: tf.Tensor
            Тензор розмірності [stages, threats, subsystems, elements].
        Z: tf.Tensor
            Тензор з даними для кожного елемента, які мають бути відображені в центрі квадрата.
        stage: int
            Індекс стадії катастрофи (0: before, 1: during, 2: after).
        criterion_name: str
            Назва критерію для заголовку графіка.
        threat_level_labels: list
            Підписи для рівнів загроз.
    """
    stage_data = S[stage].numpy()  # Дані для конкретної стадії
    Z_data = Z[stage].numpy()  # Числові значення для елементів на цій стадії
    num_threats = stage_data.shape[0]
    num_subsystems = stage_data.shape[1]
    subsystems_labels = ['Генерація', 'Транспортування', 'Споживання']

    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.matshow(stage_data, cmap="coolwarm")
    fig.colorbar(cax)

    ax.set_title(f"Теплова карта для {criterion_name} (Стадія: {['До', 'Під час', 'Після'][stage]} катастрофи)")
    ax.set_xlabel("Підсистеми")
    ax.set_ylabel("Рівні загроз")

    ax.set_xticks(range(num_subsystems))
    ax.set_xticklabels(subsystems_labels)
    ax.set_yticks(range(num_threats))
    ax.set_yticklabels(threat_level_labels)

    # Додаємо числові значення для кожного квадрата
    for i in range(num_threats):
        for j in range(num_subsystems):
            ax.text(j, i, f"{Z_data[i]:.2f}", ha='center', va='center', color="black", fontsize=10)

    plt.show()

# Функція для побудови стовпчикового графіка
def plot_bar_chart(S, criterion_name, stages_labels, subsystems_labels):
    """
    Побудова стовпчикового графіка мінімальних і максимальних значень між стадіями катастрофи.
    
    Parameters:
        S: tf.Tensor
            Тензор розмірності [stages, threats, subsystems, elements].
        criterion_name: str
            Назва критерію для заголовку графіка.
        stages_labels: list
            Підписи для стадій катастрофи.
        subsystems_labels: list
            Підписи для підсистем.
    """
    num_stages = S.shape[0]
    num_subsystems = S.shape[2]

    min_vals = tf.reduce_min(S, axis=(1, 3)).numpy()  # Мінімальні значення для кожної стадії та підсистеми
    max_vals = tf.reduce_max(S, axis=(1, 3)).numpy()  # Максимальні значення для кожної стадії та підсистеми

    x = np.arange(num_stages)  # Позиції для кожної стадії
    width = 0.35  # Ширина стовпців

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, subsystem in enumerate(subsystems_labels):
        ax.bar(x - width/2 + i*width/num_subsystems, min_vals[:, i], width/num_subsystems, label=f"{subsystem} Мін.")
        ax.bar(x + width/2 + i*width/num_subsystems, max_vals[:, i], width/num_subsystems, label=f"{subsystem} Макс.")

    ax.set_title(f"Порівняння Min/Max для {criterion_name}")
    ax.set_xlabel("Стадії катастрофи")
    ax.set_ylabel("Значення S")
    ax.set_xticks(x)
    ax.set_xticklabels(stages_labels)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# Оновлена функція для запуску експериментів з новими графіками
def run_experiments_with_new_charts(T, Z, criterion_name):
    """
    Запуск експериментів з тепловими картами і стовпчиковими графіками.
    """
    stages = ['before', 'during', 'after']
    stage_factors = {'before': 0.0, 'during': 0.5, 'after': 0.8}
    subsystems_labels = ['Генерація', 'Транспортування', 'Споживання']
    stages_labels = ['До катастрофи', 'Під час катастрофи', 'Після катастрофи']
    threat_level_labels = [f"Загроза {i+1}" for i in range(Z.shape[0])]

    S_total = []
    for stage in stages:
        modified_T = modify_tensor(T, stage, stage_factors[stage])
        S_stage = calculate_s_tensor(modified_T, Z)
        S_total.append(S_stage)

    S_total = tf.stack(S_total)  # [stages, threats, subsystems, elements]

    # Побудова теплових карт для кожної стадії
    for i, stage in enumerate(stages):
        plot_heatmap(S_total, Z, i, criterion_name, threat_level_labels)

    # Побудова стовпчикового графіка
    plot_bar_chart(S_total, criterion_name, stages_labels, subsystems_labels)

# Запуск експериментів
run_disaster_experiment(T_1, Z_1, "Критерій 1 (Функціональність)")
run_disaster_experiment(T_2, Z_2, "Критерій 2 (Здатність до відновлення)")
run_disaster_experiment(T_3, Z_3, "Критерій 3 (Стійкість до загроз)")

# Запуск експерименту Min/Max - Threat Level
run_min_max_threat_experiment()

# Запуск експерименту з побудовою 3D поверхонь для всіх критеріїв
run_3d_surface_for_all_criteria()

# Використання функцій для нових графіків
run_experiments_with_new_charts(T_1, Z_1, "Критерій 1 (Функціональність)")
run_experiments_with_new_charts(T_2, Z_2, "Критерій 2 (Здатність до відновлення)")
run_experiments_with_new_charts(T_3, Z_3, "Критерій 3 (Стійкість до загроз)")