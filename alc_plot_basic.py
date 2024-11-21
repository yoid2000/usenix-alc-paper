import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_basic_alc(als):
    # Define the range for pcc_base
    pcc_base_values = np.linspace(0, 1.0, 100)

    # Define the alc values
    alc_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the curves for each alc value
    for alc in alc_values:
        pcc_atk_values = [als.pccatk_from_pccbase_alc(pcc_base, alc) for pcc_base in pcc_base_values]
        filtered_pcc_base_values = [pcc_base for pcc_base, pcc_atk in zip(pcc_base_values, pcc_atk_values) if pcc_atk < 1.0]
        filtered_pcc_atk_values = [pcc_atk for pcc_atk in pcc_atk_values if pcc_atk < 1.0]
        ax.plot(filtered_pcc_base_values, filtered_pcc_atk_values, label=f'ALC = {alc}', linewidth=3)

    # Set the labels with larger font size
    ax.set_xlabel('PCC Base', fontsize=16)
    ax.set_ylabel('PCC Attack', fontsize=18)

    # Set the tick parameters with larger font size
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add a legend
    ax.legend()
    ax.grid(True)

    # Create an inset plot
    pcc_base_values = np.linspace(0.95, 1.0, 100)
    ax_inset = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.48, 0.15, 0.30, 0.30), bbox_transform=ax.transAxes, loc='upper left')
    for alc in alc_values:
        pcc_atk_values = [als.pccatk_from_pccbase_alc(pcc_base, alc) for pcc_base in pcc_base_values]
        filtered_pcc_base_values = [pcc_base for pcc_base, pcc_atk in zip(pcc_base_values, pcc_atk_values) if pcc_atk < 1.0]
        filtered_pcc_atk_values = [pcc_atk for pcc_atk in pcc_atk_values if pcc_atk < 1.0]
        ax_inset.plot(filtered_pcc_base_values, filtered_pcc_atk_values, label=f'ALC = {alc}', linewidth=3)

    # Set the tick parameters for the inset plot
    ax_inset.tick_params(axis='both', which='major', labelsize=10)

    # Remove the legend from the inset plot
    ax_inset.legend().set_visible(False)

    # Show the plot
    plt.tight_layout()
    return plt

# Example usage
# Assuming als is an object with the method pccatk_from_pccbase_alc
# plot = plot_basic_alc(als)
# plot.show()