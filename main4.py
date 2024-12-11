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
# Функція для обчислення тензора S
def calculate_s_tensor(T, Z, functional_dependency):
    """
    Розрахунок матриць S для заданого критерію T та вектора загроз Z.
    """
    S = []
    for z in Z:
        S.append(tf.round(functional_dependency(T, z) * 100) / 100)  # Округлення до двох знаків після крапки
    return tf.stack(S)  # Повертаємо 3D-тензор, де кожен шар відповідає одній загрозі

# Приклад функціональної залежності
def functional_linear(T, z):
    """Лінійна залежність від Z."""
    return T * z

def functional_quadratic(T, z):
    """Квадратична залежність від Z."""
    return T * (z ** 2)

def functional_custom(T, z):
    """Кастомна залежність, наприклад, з нелінійною комбінацією."""
    return T * tf.sqrt(z) + (1 - z) * tf.sin(T)

# Функція для експериментів із задаванням залежності
def run_disaster_experiment_with_functional_dependency(T, Z, criterion_name, functional_dependency):
    """
    Експеримент із моделювання сценаріїв катастроф для одного критерію із використанням залежності.
    """
    stages = ['before', 'during', 'after']
    stage_factors = {'before': 0.0, 'during': 0.5, 'after': 0.8}
    subsystems_labels = ['Генерація', 'Транспортування', 'Споживання']
    stages_labels = ['До катастрофи', 'Під час катастрофи', 'Після катастрофи']

    S_total = []
    for stage in stages:
        modified_T = modify_tensor(T, stage, stage_factors[stage])
        S_stage = calculate_s_tensor(modified_T, Z, functional_dependency)
        S_total.append(S_stage)

    S_total = tf.stack(S_total)  # [stages, threats, subsystems]
    plot_min_max(S_total, criterion_name, subsystems_labels, stages_labels)

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
def run_3d_surface_for_all_criteria(functional_dependency):
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
        S = calculate_s_tensor(T, Z, functional_dependency)

        # Побудова 3D поверхні для поточного критерію
        plot_3d_surface_for_threat_level(S, Z, criterion_name)

# Приклад запуску із різними функціональними залежностями
run_disaster_experiment_with_functional_dependency(T_1, Z_1, "Критерій 1 (Лінійна залежність)", functional_linear)
run_disaster_experiment_with_functional_dependency(T_2, Z_2, "Критерій 2 (Квадратична залежність)", functional_quadratic)
run_disaster_experiment_with_functional_dependency(T_3, Z_3, "Критерій 3 (Кастомна залежність)", functional_custom)

# Запуск експерименту з побудовою 3D поверхонь для всіх критеріїв
run_3d_surface_for_all_criteria(functional_linear)
run_3d_surface_for_all_criteria(functional_quadratic)