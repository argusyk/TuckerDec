# import numpy as np
# import tensorflow as tf
# from scipy.io.matlab import loadmat
# from ktensor import KruskalTensor

""" # Load sensory bread data (http://www.models.life.ku.dk/datasets)
mat = loadmat('data\bread\brod.mat')
X = mat['X'].reshape([10,11,8])

# Build ktensor and learn CP decomposition using ALS with specified optimizer
T = KruskalTensor(X.shape, rank=3, regularize=1e-6, init='nvecs', X_data=X)
X_predict = T.train_als(X, tf.train.AdadeltaOptimizer(0.05), epochs=20000)

# Save reconstructed tensor to file
np.save('X_predict.npy', X_predict) """

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# Створення функції для обчислення тензора S
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
    - stage: 'before', 'during', 'after'
    - factor: значення, що поступово змінюється між етапами.
    """
    if stage == 'before':
        return T  # Без змін
    elif stage == 'during':
        return T * (1 - factor)  # Зменшення функціональності залежно від фактору
    elif stage == 'after':
        return T * (0.5 + 0.5 * factor)  # Відновлення функціональності
    else:
        raise ValueError("Unknown stage. Use 'before', 'during', or 'after'.")

# Функція для візуалізації тензорів
def plot_tensor(tensor, title):
    """
    Візуалізація 3D-тензора у вигляді матриць.
    """
    num_matrices = tensor.shape[0]
    fig, axes = plt.subplots(1, num_matrices, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    for i in range(num_matrices):
        ax = axes[i]
        ax.matshow(tensor[i].numpy(), cmap='viridis', alpha=0.8)
        for (j, k), val in np.ndenumerate(tensor[i].numpy()):
            ax.text(k, j, f"{val:.2f}", ha='center', va='center', color='white')
        ax.set_title(f"Загроза {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

# Функція для проведення експерименту
def disaster_scenario_experiment(T, Z, criterion_name):
    """
    Експеримент із моделювання сценаріїв катастроф.
    """
    stages = ['before', 'during', 'after']
    for stage in stages:
        # Встановлення фактору залежно від стадії
        if stage == 'before':
            factor = 0.0
        elif stage == 'during':
            factor = 0.5
        elif stage == 'after':
            factor = 0.8
        else:
            raise ValueError("Unknown stage. Use 'before', 'during', or 'after'.")

        # Модифікація матриці T відповідно до стадії
        modified_T = modify_tensor(T, stage, factor)
        # Обчислення тензора S
        S = calculate_s_tensor(modified_T, Z)
        # Візуалізація тензора S
        plot_tensor(S, f"{criterion_name} - {stage.capitalize()} Disaster")

# Функція для обчислення та відображення мінімальних та максимальних значень тензора S
def plot_min_max(S_total, criterion_name):
    """
    Побудова графіків мінімум та максимум тензора S на різних стадіях для різних підсистем.
    """
    # Обчислення мінімумів та максимумів по загрозах (axis=1)
    min_S = tf.reduce_min(S_total, axis=1).numpy()  # [criteria, subsystems, stages]
    max_S = tf.reduce_max(S_total, axis=1).numpy()  # [criteria, subsystems, stages]

    stages = ['До катастрофи', 'Під час катастрофи', 'Після катастрофи']
    subsystems = ['Генерація', 'Транспортування', 'Споживання']

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle(f"Мінімум та Максимум тензора S для {criterion_name}", fontsize=16)

    for j in range(3):  # Для кожної підсистеми
        ax = axes[j]
        ax.plot(stages, min_S[j], label='Мінімум', marker='o')
        ax.plot(stages, max_S[j], label='Максимум', marker='s')
        ax.set_title(f"Підсистема: {subsystems[j]}")
        ax.set_xlabel("Стадія катастрофи")
        ax.set_ylabel("Значення S")
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Проведення експерименту для кожного критерію
def run_all_experiments():
    """
    Проведення експериментів та побудова графіків мінімумів та максимумів.
    """
    # Обчислення початкових тензорів S
    S_1_before = calculate_s_tensor(T_1, Z_1)
    S_2_before = calculate_s_tensor(T_2, Z_2)
    S_3_before = calculate_s_tensor(T_3, Z_3)

    # Об'єднання тензорів S в загальний тензор стійкості
    # Формат: [criteria, threats, subsystems, stages]
    S_total = tf.stack([S_1_before, S_2_before, S_3_before])  # Початковий тензор

    # Проведення експерименту (до, під час, після катастрофи)
    stages = ['before', 'during', 'after']
    criterion_names = ["Criterion 1 (Infrastructure Functionality)",
                       "Criterion 2 (Recovery Ability)",
                       "Criterion 3 (Resilience to Threats)"]
    Ts = [T_1, T_2, T_3]
    Zs = [Z_1, Z_2, Z_3]

    for i in range(3):
        T = Ts[i]
        Z = Zs[i]
        criterion_name = criterion_names[i]
        disaster_scenario_experiment(T, Z, criterion_name)

    # Обчислення мінімумів та максимумів після експерименту
    # Для цього потрібно зберегти всі тензори S під час експерименту
    # Оскільки функція disaster_scenario_experiment не повертає значень, змінімо її

def disaster_scenario_experiment_collect(T, Z, criterion_name, S_total):
    """
    Експеримент із моделювання сценаріїв катастроф з накопиченням тензора S.
    """
    stages = ['before', 'during', 'after']
    stage_factors = {'before': 0.0, 'during': 0.5, 'after': 0.8}
    for stage in stages:
        # Отримання фактору для поточної стадії
        factor = stage_factors[stage]
        # Модифікація матриці T відповідно до стадії
        modified_T = modify_tensor(T, stage, factor)
        # Обчислення тензора S
        S = calculate_s_tensor(modified_T, Z)
        # Додавання до загального тензора S_total
        # Формат: [criteria, threats, subsystems, stages]
        # Додаємо по загрозах для кожної стадії
        # Це може бути організовано різними способами. Одним із варіантів:
        # Зберігаємо стани після кожної стадії окремо
        # Для спрощення, зберігаємо результати окремо
        for l in range(S.shape[0]):
            S_total = tf.concat([S_total, tf.expand_dims(S[l], axis=0)], axis=0)
    return S_total

# Обчислення та відображення мінімумів та максимумів
def plot_min_max_all(S_total, criterion_names):
    """
    Побудова графіків мінімум та максимум тензора S на різних стадіях для різних підсистем.
    """
    # Припустимо, що S_total має формат [criteria * stages, threats, subsystems, stages]
    # Для трьох критеріїв та трьох стадій, кількість записів буде 9
    num_criteria = len(criterion_names)
    num_stages = 3  # before, during, after
    num_subsystems = 3
    num_threats = 4

    # Перетворюємо S_total на [criteria, stages, threats, subsystems, stages]
    # Але це може бути складно, тому спростимо

    # Для кожного критерію, збираємо всі тензори S для всіх стадій
    for i in range(num_criteria):
        # Для кожного критерію, мінімум та максимум по загрозах
        # Зібрати всі S_i^l для цього критерію
        # У нашому прикладі, припустимо, що S_total зберігає всі стадії разом
        # Тобто, кожна стадія має окремий набір загроз
        # Наприклад, для T1: S_total[0:4], S_total[9:13], S_total[18:22]

        # Індекс старту для цього критерію
        start_idx = i * num_stages * num_threats
        # Індекс кінця
        end_idx = start_idx + num_stages * num_threats
        # Вибрати відповідні тензори
        S_i = S_total[start_idx:end_idx]  # [stages * threats, subsystems, stages]

        # Створення масиву для зберігання мінімумів та максимумів
        min_vals = np.zeros((num_subsystems, num_stages))
        max_vals = np.zeros((num_subsystems, num_stages))

        for stage in range(num_stages):
            for subsystem in range(num_subsystems):
                # Для поточної стадії та підсистеми, збираємо всі загрози
                # Кожна стадія має num_threats загроз
                threat_start = stage * num_threats
                threat_end = threat_start + num_threats
                # Вибір S_i[l, subsystem, stage] для l in загрози
                S_stage = S_i[threat_start:threat_end, subsystem, stage].numpy()
                # Обчислення мінімуму та максимуму
                min_vals[subsystem, stage] = np.min(S_stage)
                max_vals[subsystem, stage] = np.max(S_stage)

        # Побудова графіку для цього критерію
        stages_labels = ['До катастрофи', 'Під час катастрофи', 'Після катастрофи']
        subsystems_labels = ['Генерація', 'Транспортування', 'Споживання']

        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        fig.suptitle(f"Мінімум та Максимум тензора S для {criterion_names[i]}", fontsize=16)

        for j in range(num_subsystems):
            ax = axes[j]
            ax.plot(stages_labels, min_vals[j], label='Мінімум', marker='o')
            ax.plot(stages_labels, max_vals[j], label='Максимум', marker='s')
            ax.set_title(f"Підсистема: {subsystems_labels[j]}")
            ax.set_xlabel("Стадія катастрофи")
            ax.set_ylabel("Значення S")
            ax.legend()
            ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# Оновлена функція для проведення експерименту та накопичення S_total
def disaster_scenario_experiment_collect(T, Z, criterion_name, S_total, num_stages=3, num_threats=4):
    """
    Експеримент із моделювання сценаріїв катастроф з накопиченням тензора S.
    """
    stages = ['before', 'during', 'after']
    stage_factors = {'before': 0.0, 'during': 0.5, 'after': 0.8}
    for stage in stages:
        # Отримання фактору для поточної стадії
        factor = stage_factors[stage]
        # Модифікація матриці T відповідно до стадії
        modified_T = modify_tensor(T, stage, factor)
        # Обчислення тензора S
        S = calculate_s_tensor(modified_T, Z)
        # Додавання до загального тензора S_total
        S_total = tf.concat([S_total, S], axis=0)
    return S_total

# Основна функція для запуску експериментів
def run_all_experiments_with_min_max():
    """
    Проведення експериментів та побудова графіків мінімумів та максимумів.
    """
    criterion_names = ["Criterion 1 (Infrastructure Functionality)",
                       "Criterion 2 (Recovery Ability)",
                       "Criterion 3 (Resilience to Threats)"]
    Ts = [T_1, T_2, T_3]
    Zs = [Z_1, Z_2, Z_3]

    # Ініціалізація порожнього тензора S_total
    # Формат: [criteria * stages * threats, subsystems, stages]
    S_total = tf.constant([], dtype=tf.float32)
    S_total = tf.reshape(S_total, (0, 3, 3))  # Початковий порожній тензор

    # Проведення експериментів та накопичення S_total
    for i in range(3):
        T = Ts[i]
        Z = Zs[i]
        criterion_name = criterion_names[i]
        S_total = disaster_scenario_experiment_collect(T, Z, criterion_name, S_total)

    # Виведення всіх тензорів S після експерименту
    print("\nЗагальний тензор S після експерименту:")
    print(S_total.numpy())

    # Обчислення мінімумів та максимумів
    plot_min_max_all(S_total, criterion_names)

# Запуск всіх експериментів та побудова графіків мінімумів та максимумів
run_all_experiments_with_min_max()

# Анімація змін тензорів (опціонально)
def animate_tensor_dynamics(T, Z, criterion_name):
    """
    Анімація змін тензорів на різних стадіях катастрофи.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{criterion_name} Dynamics", fontsize=16)

    # Підготовка підзагроз
    threat_indices = np.arange(len(Z.numpy())) + 1

    # Створення початкової матриці
    initial_S = calculate_s_tensor(T, Z).numpy()
    images = [axes[i].matshow(initial_S[i], cmap='viridis', alpha=0.8) for i in range(len(Z))]
    for i, ax in enumerate(axes):
        ax.set_title(f"Загроза {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])

    def update(frame):
        # Визначення стадії та фактору
        if frame < 50:
            stage = 'before'
            factor = 0.0
        elif frame < 100:
            stage = 'during'
            factor = (frame - 50) / 50  # Зростання від 0 до 1
        else:
            stage = 'after'
            factor = (frame - 100) / 50  # Зростання від 0 до 1

        # Модифікація матриці T відповідно до стадії
        modified_T = modify_tensor(T, stage, factor)
        # Обчислення тензора S
        S = calculate_s_tensor(modified_T, Z).numpy()

        # Оновлення зображень
        for i, img in enumerate(images):
            img.set_data(S[i])
            # Оновлення тексту
            for (j, k), val in np.ndenumerate(S[i]):
                axes[i].text(k, j, f"{val:.2f}", ha='center', va='center', color='white')
        # Оновлення заголовка
        fig.suptitle(f"{criterion_name} - {stage.capitalize()} Disaster (Frame: {frame})", fontsize=16)
        return images

    ani = FuncAnimation(fig, update, frames=150, interval=100, blit=False)
    plt.show()

# Приклад анімації для одного критерію
# Uncomment the following lines to run animation for each criterion
# animate_tensor_dynamics(T_1, Z_1, "Criterion 1 (Infrastructure Functionality)")
# animate_tensor_dynamics(T_2, Z_2, "Criterion 2 (Recovery Ability)")
# animate_tensor_dynamics(T_3, Z_3, "Criterion 3 (Resilience to Threats)")