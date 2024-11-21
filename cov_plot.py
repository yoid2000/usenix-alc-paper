import numpy as np
import matplotlib.pyplot as plt
import alscore
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def p_atk_from_p_base_cov(als, p_base, cov, alc=0.5):
    pcc_base = als.pcc(p_base, cov)
    pcc_atk = als.pccatk_from_pccbase_alc(pcc_base, alc)
    p_atk = als.prec_from_pcc_cov(pcc_atk, cov)
    return p_atk if p_atk <= 1.0 else None

def plot_coverage_alc(als):
    # Define the range for pcc_base
    prec_base_values = np.linspace(0, 1.0, 100)

    cov_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the curves for each alc value
    for cov in cov_values:
        p_atk_values = [p_atk_from_p_base_cov(als, p_base, cov, alc=0.5) for p_base in prec_base_values]
        filtered_p_base_values = [p_base for p_base, p_atk in zip(prec_base_values, p_atk_values) if p_atk is not None]
        filtered_p_atk_values = [p_atk for p_atk in p_atk_values if p_atk is not None]
        ax.plot(filtered_p_base_values, filtered_p_atk_values, label=f'Cov = {cov}', linewidth=3)

    # Set the labels with larger font size
    ax.set_xlabel('Precision Base', fontsize=16)
    ax.set_ylabel('Precision Attack', fontsize=18)

    # Set the tick parameters with larger font size
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add a legend
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    return plt


# Example usage
# Assuming als is an object with the method pccatk_from_pccbase_alc
# plot = plot_basic_alc(als)
# plot.show()