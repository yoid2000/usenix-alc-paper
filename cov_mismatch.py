import numpy as np
import matplotlib.pyplot as plt

def generate_points(als, c_ratio, p_base, num_points=1000):
    points = []
    for _ in range(num_points):
        p_attack = p_base
        c_attack = np.random.uniform(0.001, 1.0)
        c_base = c_attack * c_ratio
        alc = als.alscore(p_base=p_base, c_base=c_base, p_attack=p_attack, c_attack=c_attack)
        points.append((c_attack, alc))
    return points

def plot_alscore_scatter(als):
    c_ratios = [1/1.1, 0.5, 0.1, 0.05]
    colors = ['red', 'blue', 'green', 'orange']
    #labels = [f'c_base/c_attack = {c_ratio}' for c_ratio in c_ratios]
    labels = [f'c_attack/c_base = {1/c_ratio}' for c_ratio in c_ratios]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Generate points and plot for p_base = 0.5
    p_base = 0.5
    all_alc_values_1 = []
    for c_ratio, color, label in zip(c_ratios, colors, labels):
        points = generate_points(als, c_ratio, p_base)
        c_attack_values, alc_values = zip(*points)
        all_alc_values_1.extend(alc_values)
        ax1.scatter(c_attack_values, alc_values, color=color, label=label, alpha=0.6)
    ax1.set_xlabel('Attack Coverage', fontsize=14)
    ax1.set_ylabel('ALC', fontsize=14)
    ax1.set_title(f'Baseline and Attack Precision = {p_base}', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # Generate points and plot for p_base = 0.9
    p_base = 0.95
    all_alc_values_2 = []
    for c_ratio, color, label in zip(c_ratios, colors, labels):
        points = generate_points(als, c_ratio, p_base)
        c_attack_values, alc_values = zip(*points)
        all_alc_values_2.extend(alc_values)
        ax2.scatter(c_attack_values, alc_values, color=color, label=label, alpha=0.6)
    ax2.set_xlabel('Attack Coverage', fontsize=14)
    ax2.set_ylabel('ALC', fontsize=14)
    ax2.set_title(f'Baseline and Attack Precision = {p_base}', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    # Determine the combined y-axis range
    combined_y_min = min(min(all_alc_values_1), min(all_alc_values_2))
    combined_y_max = max(max(all_alc_values_1), max(all_alc_values_2))

    # Set the same y-axis range for both plots
    ax1.set_ylim(combined_y_min, combined_y_max)
    ax2.set_ylim(combined_y_min, combined_y_max)

    plt.tight_layout()
    return plt