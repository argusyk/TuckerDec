import numpy as np
import matplotlib.pyplot as plt

# Імітація тензора S для прикладу
def simulate_tensor(dimensions, steps):
    """
    Генерує тензор S для заданих розмірностей та кількості кроків.
    """
    return np.random.rand(steps, *dimensions)

# Функція для імітації зміни загрози
def simulate_threat_dynamics(steps):
    """
    Імітує зміну загрози від 0 до 1 у заданій кількості кроків.
    """
    return np.linspace(0, 1, steps)

# Модифікована функція для побудови мінімуму та максимуму
def plot_min_max_with_threat(S, threat_levels, labels=None):
    """
    Будує графіки мінімуму та максимуму тензора S для різних стадій 
    з урахуванням рівнів загрози.
    
    Parameters:
        S: np.ndarray
            Тензор значень (з розмірністю [steps, dim1, dim2]).
        threat_levels: np.ndarray
            Масив рівнів загрози, довжина якого відповідає кількості кроків.
        labels: list
            Список підсистем для позначення на графіку.
    """
    steps = S.shape[0]
    subsystems = S.shape[1]
    
    if labels is None:
        labels = [f"Subsystem {i+1}" for i in range(subsystems)]
    
    plt.figure(figsize=(12, 6))
    for i in range(subsystems):
        min_values = np.min(S[:, i, :], axis=1)
        max_values = np.max(S[:, i, :], axis=1)
        
        plt.plot(threat_levels, min_values, label=f"{labels[i]} Min", linestyle='--')
        plt.plot(threat_levels, max_values, label=f"{labels[i]} Max", linestyle='-')
    
    plt.xlabel("Threat Level")
    plt.ylabel("Values (Min / Max)")
    plt.title("Min and Max Values of Tensor S Across Subsystems with Threat Dynamics")
    plt.legend()
    plt.grid(True)
    plt.show()

# Використання функцій
steps = 100
dimensions = (3, 5)  # 3 підсистеми, кожна має 5 елементів

# Генерація даних
S = simulate_tensor(dimensions, steps)
threat_levels = simulate_threat_dynamics(steps)
labels = ["Subsystem A", "Subsystem B", "Subsystem C"]

# Побудова графіка
plot_min_max_with_threat(S, threat_levels, labels)
