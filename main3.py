import numpy as np
import matplotlib.pyplot as plt

# Генерація даних тензорів T і загроз Z
def generate_dynamic_threats(num_criteria, num_threats, steps):
    """
    Генерує змінні рівні загрози для кожного критерію в динаміці.
    """
    return np.array([np.linspace(0, 1, steps) for _ in range(num_threats * num_criteria)]).reshape(num_criteria, num_threats, steps)

# Розрахунок тензорів S для кожного кроку динаміки загрози
def calculate_dynamic_S(T, Z_dynamic):
    """
    Розраховує тензор S для кожного кроку в часі залежно від динамічної загрози.
    """
    num_criteria, num_threats, steps = Z_dynamic.shape
    S_dynamic = np.zeros((steps, *T.shape))  # Розмір: [steps, num_criteria, 3, 3]
    
    for step in range(steps):
        for i in range(num_criteria):
            for l in range(num_threats):
                S_dynamic[step, i] += T[i] * Z_dynamic[i, l, step]
    return S_dynamic

# Модифікована функція для побудови графіка
def plot_min_max_dynamic(S_dynamic, labels=None):
    """
    Будує графік мінімуму та максимуму тензора S на різних етапах динаміки загрози.
    """
    steps, num_criteria, _, _ = S_dynamic.shape
    min_values = np.min(S_dynamic, axis=(2, 3))  # Мінімальні значення для кожного критерію на кожному етапі
    max_values = np.max(S_dynamic, axis=(2, 3))  # Максимальні значення для кожного критерію на кожному етапі
    
    if labels is None:
        labels = [f"Criterion {i+1}" for i in range(num_criteria)]

    plt.figure(figsize=(14, 8))
    for i in range(num_criteria):
        plt.plot(range(steps), min_values[:, i], label=f"{labels[i]} Min", linestyle='--')
        plt.plot(range(steps), max_values[:, i], label=f"{labels[i]} Max", linestyle='-')

    plt.xlabel("Time Step")
    plt.ylabel("Values (Min / Max)")
    plt.title("Dynamic Min and Max Values of Tensor S")
    plt.legend()
    plt.grid(True)
    plt.show()

# Дані тензорів для прикладу
T = np.array([
    [[0.9, 0.8, 0.7], [0.85, 0.8, 0.75], [0.7, 0.6, 0.5]],  # T1
    [[0.7, 0.6, 0.5], [0.6, 0.55, 0.5], [0.4, 0.3, 0.2]],   # T2
    [[0.8, 0.7, 0.6], [0.75, 0.65, 0.55], [0.5, 0.4, 0.3]], # T3
])

# Динаміка загроз
num_criteria = 3
num_threats = 4
steps = 100
Z_dynamic = generate_dynamic_threats(num_criteria, num_threats, steps)

# Розрахунок тензора S у динаміці
S_dynamic = calculate_dynamic_S(T, Z_dynamic)

# Побудова графіка
labels = ["Infrastructure Functionality", "Recovery Ability", "Resistance to External Threats"]
plot_min_max_dynamic(S_dynamic, labels)
