import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialization of T matrices for three criteria
T_1 = tf.constant([[0.9, 0.8, 0.7],
                   [0.85, 0.8, 0.75],
                   [0.7, 0.6, 0.5]], dtype=tf.float32)

T_2 = tf.constant([[0.7, 0.6, 0.5],
                   [0.6, 0.55, 0.5],
                   [0.4, 0.3, 0.2]], dtype=tf.float32)

T_3 = tf.constant([[0.8, 0.7, 0.6],
                   [0.75, 0.65, 0.55],
                   [0.5, 0.4, 0.3]], dtype=tf.float32)

# Initialization of threat vectors Z for three criteria
Z_1 = tf.constant([0.9, 0.8, 0.7, 0.6], dtype=tf.float32)
Z_2 = tf.constant([0.85, 0.75, 0.65, 0.55], dtype=tf.float32)
Z_3 = tf.constant([0.8, 0.7, 0.6, 0.5], dtype=tf.float32)

# Function to calculate the S tensor
def calculate_s_tensor(T, Z):
    """
    Calculate S matrices for a given T criterion and threat vector Z.
    """
    S = []
    for z in Z:
        S.append(tf.round(T * z * 100) / 100)  # Round to two decimal places
    return tf.stack(S)  # Return a 3D tensor, where each layer corresponds to one threat

# Function to model catastrophic changes in tensor T
def modify_tensor(T, stage, factor):
    """
    Smoothly modify the T matrix according to the disaster stage.
    """
    if stage == 'before':
        return T
    elif stage == 'during':
        return T * (1 - factor)
    elif stage == 'after':
        return T * (0.5 + 0.5 * factor)
    else:
        raise ValueError("Unknown stage. Use 'before', 'during', or 'after'.")

# Function to calculate and display the minimum and maximum values of the S tensor
def plot_min_max(S, criterion_name, subsystems_labels, stages_labels):
    """
    Plot graphs of minimum and maximum for the S tensor.
    """
    num_stages = S.shape[0]
    num_subsystems = S.shape[2]

    min_vals = np.min(S.numpy(), axis=1)
    max_vals = np.max(S.numpy(), axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(num_subsystems):
        ax.plot(stages_labels, min_vals[:, i], label=f"{subsystems_labels[i]} (Min)", linestyle='--', marker='o')
        ax.plot(stages_labels, max_vals[:, i], label=f"{subsystems_labels[i]} (Max)", linestyle='-', marker='s')
    
    ax.set_title(f"Minimum and Maximum for {criterion_name}")
    ax.set_xlabel("Disaster Stage")
    ax.set_ylabel("S Values")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Main function to run experiments
def run_disaster_experiment(T, Z, criterion_name):
    """
    Experiment modeling disaster scenarios for a single criterion.
    """
    stages = ['before', 'during', 'after']
    stage_factors = {'before': 0.0, 'during': 0.5, 'after': 0.8}
    subsystems_labels = ['Generation', 'Transportation', 'Consumption']
    stages_labels = ['Before Disaster', 'During Disaster', 'After Disaster']

    S_total = []
    for stage in stages:
        modified_T = modify_tensor(T, stage, stage_factors[stage])
        S_stage = calculate_s_tensor(modified_T, Z)
        S_total.append(S_stage)

    S_total = tf.stack(S_total)  # [stages, threats, subsystems]
    plot_min_max(S_total, criterion_name, subsystems_labels, stages_labels)

# Function for plotting Min/Max - Threat Level
def plot_min_max_threat_level(S, Z, criterion_name):
    """
    Plotting the graph of minimum and maximum S values based on threat levels Z.
    
    Parameters:
        S: tf.Tensor
            Tensor with dimensions [threat_levels, subsystems, ...].
        Z: tf.Tensor
            Vector of threat levels.
        criterion_name: str
            Criterion name for the plot title.
    """
    num_threats = Z.shape[0]
    num_subsystems = S.shape[1]

    min_vals = tf.reduce_min(S, axis=2).numpy()  # Minimum values across elements of each subsystem
    max_vals = tf.reduce_max(S, axis=2).numpy()  # Maximum values across elements of each subsystem

    subsystems_labels = ['Generation', 'Transportation', 'Consumption']

    plt.figure(figsize=(10, 6))
    for i in range(num_subsystems):
        plt.plot(Z.numpy(), min_vals[:, i], label=f'{subsystems_labels[i]} Minimum', linestyle='--', marker='o')
        plt.plot(Z.numpy(), max_vals[:, i], label=f'{subsystems_labels[i]} Maximum', linestyle='-', marker='s')

    plt.xlabel('Threat Level')
    plt.ylabel('Values (Min / Max)')
    plt.title(f"Min/Max values for {criterion_name} depending on threat levels")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function usage
def run_min_max_threat_experiment():
    """
    Run an experiment with plotting Min/Max - Threat Level graphs.
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

        # Calculate the S tensor for each criterion
        S = calculate_s_tensor(T, Z)

        # Plot Min/Max - Threat Level graphs
        plot_min_max_threat_level(S, Z, criterion_name)

# Function for plotting 3D surfaces for S values depending on threat levels
def plot_3d_surface_for_threat_level(S, Z, criterion_name):
    """
    Plotting 3D surface of S values depending on threat levels Z for each criterion.
    
    Parameters:
        S: tf.Tensor
            Tensor with dimensions [threat_levels, subsystems, elements_per_subsystem].
        Z: tf.Tensor
            Vector of threat levels.
        criterion_name: str
            Criterion name for the plot title.
    """
    num_threats = Z.shape[0]
    num_subsystems = S.shape[1]
    num_elements = S.shape[2]

    # Create a grid for the 3D plot
    subsystems_labels = ['Generation', 'Transportation', 'Consumption']
    elements_labels = ['Post-disaster', 'During disaster', 'Pre-disaster']

    X = np.arange(num_subsystems)  # Subsystems axis (0, 1, 2)
    Y = np.arange(num_elements)   # Elements in subsystems
    X, Y = np.meshgrid(X, Y)

    # Create a figure for the 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D surface for each threat level
    for threat_level in range(num_threats):
        Z_surface = S[threat_level].numpy()  # Get values for the current threat level
        ax.plot_surface(X, Y, Z_surface, alpha=0.7, cmap='viridis', edgecolor='k')

    ax.set_xlabel("Subsystems")
    ax.set_ylabel("Disaster Stage")
    ax.set_zlabel("S Values")
    ax.set_title(f"3D Surface for {criterion_name} at Different Threat Levels")

    # Set labels for X and Y axes
    ax.set_xticks(np.arange(num_subsystems))  # X-axis ticks (0, 1, 2)
    ax.set_xticklabels(subsystems_labels)  # X-axis labels
    ax.set_yticks(np.arange(num_elements))  # Y-axis ticks (0, 1, 2)
    ax.set_yticklabels(elements_labels)  # Y-axis labels

    plt.show()

# Function to run the experiment of building 3D surfaces for each criterion
def run_3d_surface_for_all_criteria():
    """
    Run the experiment of building 3D surfaces for each criterion.
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

        # Calculate the S tensor for each criterion
        S = calculate_s_tensor(T, Z)

        # Build the 3D surface for the current criterion
        plot_3d_surface_for_threat_level(S, Z, criterion_name)

# Function to build a heatmap of the S tensor values
def plot_heatmap(S, stage, criterion_name, threat_level_labels):
    """
    Build a heatmap of the S tensor values for the given disaster stage.
    
    Parameters:
        S: tf.Tensor
            Tensor of shape [stages, threats, subsystems, elements].
        stage: int
            Index of the disaster stage (0: before, 1: during, 2: after).
        criterion_name: str
            Criterion name for the chart title.
        threat_level_labels: list
            Labels for the threat levels.
    """
    stage_data = S[stage].numpy()  # Data for the specific stage
    num_threats = stage_data.shape[0]
    subsystems_labels = ['Generation', 'Transportation', 'Consumption']

    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.matshow(stage_data, cmap="seismic")
    fig.colorbar(cax)

    ax.set_title(f"Heatmap for {criterion_name} (Stage: {['Before', 'During', 'After'][stage]} of disaster)")
    ax.set_xlabel("Subsystems")
    ax.set_ylabel("Threat Levels")

    ax.set_xticks(range(len(subsystems_labels)))
    ax.set_xticklabels(subsystems_labels)
    ax.set_yticks(range(num_threats))
    ax.set_yticklabels(threat_level_labels)

    plt.show()


# Function to build a bar chart
def plot_bar_chart(S, criterion_name, stages_labels, subsystems_labels):
    """
    Build a bar chart of the minimum and maximum values between disaster stages.
    
    Parameters:
        S: tf.Tensor
            Tensor of shape [stages, threats, subsystems, elements].
        criterion_name: str
            Criterion name for the chart title.
        stages_labels: list
            Labels for disaster stages.
        subsystems_labels: list
            Labels for subsystems.
    """
    num_stages = S.shape[0]
    num_subsystems = S.shape[2]

    min_vals = tf.reduce_min(S, axis=(1, 3)).numpy()  # Minimum values for each stage and subsystem
    max_vals = tf.reduce_max(S, axis=(1, 3)).numpy()  # Maximum values for each stage and subsystem

    x = np.arange(num_stages)  # Positions for each stage
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, subsystem in enumerate(subsystems_labels):
        ax.bar(x - width/2 + i*width/num_subsystems, min_vals[:, i], width/num_subsystems, label=f"{subsystem} Min.")
        ax.bar(x + width/2 + i*width/num_subsystems, max_vals[:, i], width/num_subsystems, label=f"{subsystem} Max.")

    ax.set_title(f"Min/Max Comparison for {criterion_name}")
    ax.set_xlabel("Disaster Stages")
    ax.set_ylabel("S Values")
    ax.set_xticks(x)
    ax.set_xticklabels(stages_labels)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# Updated function to run experiments with new charts
def run_experiments_with_new_charts(T, Z, criterion_name):
    """
    Run experiments with heatmaps and bar charts.
    """
    stages = ['before', 'during', 'after']
    stage_factors = {'before': 0.0, 'during': 0.5, 'after': 0.8}
    subsystems_labels = ['Generation', 'Transportation', 'Consumption']
    stages_labels = ['Before disaster', 'During disaster', 'After disaster']
    threat_level_labels = [f"Threat {i+1}" for i in range(Z.shape[0])]

    S_total = []
    for stage in stages:
        modified_T = modify_tensor(T, stage, stage_factors[stage])
        S_stage = calculate_s_tensor(modified_T, Z)
        S_total.append(S_stage)

    S_total = tf.stack(S_total)  # [stages, threats, subsystems, elements]

    # Build heatmaps for each stage
    for i, stage in enumerate(stages):
        plot_heatmap(S_total, i, criterion_name, threat_level_labels)

    # Build the bar chart
    plot_bar_chart(S_total, criterion_name, stages_labels, subsystems_labels)

# Run the experiments
run_disaster_experiment(T_1, Z_1, "Criterion 1 (Functionality)")
run_disaster_experiment(T_2, Z_2, "Criterion 2 (Recovery Ability)")
run_disaster_experiment(T_3, Z_3, "Criterion 3 (Resilience to Threats)")

# Run the Min/Max Threat Level experiment
run_min_max_threat_experiment()

# Run the experiment with building 3D surfaces for all criteria
run_3d_surface_for_all_criteria()

# Using functions for new charts
run_experiments_with_new_charts(T_1, Z_1, "Criterion 1 (Functionality)")
run_experiments_with_new_charts(T_2, Z_2, "Criterion 2 (Recovery Ability)")
run_experiments_with_new_charts(T_3, Z_3, "Criterion 3 (Resilience to Threats)")
