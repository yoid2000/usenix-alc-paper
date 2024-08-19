import numpy as np
import random
import os
import matplotlib.pyplot as plt
import alscore
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pprint
pp = pprint.PrettyPrinter(indent=4)

plots_path = os.path.join('als_plots')
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

def savefigs(plt, name):
    for suffix in ['.png', '.pdf']:
        path_name = name + suffix
        out_path = os.path.join(plots_path, path_name)
        plt.savefig(out_path)

def plot_prec_cov_for_equal_pcc(out_name):
    ''' The purpose of this plot is to see how different values of prec
        and cov can have the same pcc.
    '''
    als = alscore.ALScore()
    alpha = als.get_param('cov_adjust_strength')
    print(f'for pcc = 0.5, and prec 1.0, cov = {als.cov_from_pcc_prec(0.5, 1.0)}')
    print(f'for pcc = 0.5, and cov 0.001484, prec = {als.prec_from_pcc_cov(0.5, 0.001484)}')
    print(f'for pcc = 0.5, and prec 0.6, cov = {als.cov_from_pcc_prec(0.5, 0.6)}')
    print(f'for pcc = 0.5, and cov 0.0233, prec = {als.prec_from_pcc_cov(0.5, 0.0233)}')
    print(f'for pcc = 0.5, and prec 0.5, cov = {als.cov_from_pcc_prec(0.5, 0.5)}')
    print(f'for pcc = 0.5, and cov 1.0, prec = {als.prec_from_pcc_cov(0.5, 1.0)}')
    pcc_vals = [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.001]
    ranges = [[0.0001, 0.00011], [0.00011, 0.001], [0.001, 0.01], [0.01, 0.1], [0.1, 1]]
    arrays = [np.linspace(start, end, 1000) for start, end in ranges]
    cov_base_values = np.concatenate(arrays)

    plt.figure(figsize=((6, 3.5)))
    for pcc_val in pcc_vals:
        prec_values = [als.prec_from_pcc_cov(pcc_val, cov_value) for cov_value in cov_base_values]
        prec_cov_pairs = [(prec, cov) for prec, cov in zip(prec_values, cov_base_values)]
        prec_cov_pairs = sorted(prec_cov_pairs, key=lambda x: x[0])
        prec_cov_pairs = [(prec, cov) for prec, cov in prec_cov_pairs if prec <= 1.0]
        prec_values, cov_values = zip(*prec_cov_pairs)
        plt.scatter(cov_values, prec_values, label=f'PCC = {pcc_val}', s=1)
    plt.xscale('log')
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Coverage', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.text(0.05, 0.98, f'alpha = {alpha}, Cmin = 0.0001', ha='left', va='top', fontsize=9, transform=plt.gca().transAxes)
    plt.legend(scatterpoints=1, markerscale=7, handletextpad=0.5, labelspacing=0.5, fontsize='small', loc='lower center')
    plt.tight_layout()
    savefigs(plt, out_name)

def run_pcc_checks(als, p_base, c_base, pcc_base):
    if c_base <= 0.0001:
        return
    p_base_test = round(als.prec_from_pcc_cov(pcc_base, c_base),3)
    if round(p_base,3) != p_base_test:
        print(f'Error: prec_from_pcc_cov({pcc_base}, {c_base})')
        print(f'Expected: {round(p_base,3)}, got: {p_base_test}')
        quit()
    c_base_test = round(als.cov_from_pcc_prec(pcc_base, p_base),3)
    if round(c_base,3) != c_base_test:
        print(f'Error: cov_from_pcc_prec({pcc_base}, {p_base})')
        print(f'Expected: {round(c_base,3)}, got: {c_base_test}')
        quit()

def do_als_test(als, p_base, c_base, increase, c_attack):
    print('------------------------------------')
    p_attack = p_base + increase * (1.0 - p_base)
    print(f'Base precision: {p_base}, base coverage: {c_base}\nattack precision: {p_attack}, attack coverage: {c_attack}')
    print(f'prec increase: {increase}')
    pcc_atk = als.pcc(prec=p_attack, cov=c_attack)
    print(f'pcc_atk: {pcc_atk}')
    pcc_base = als.pcc(prec=p_base, cov=c_base)
    print(f'pcc_base: {pcc_base}')
    run_pcc_checks(als, p_base, c_base, pcc_base)
    print(f'ALS: {round(als.alscore(p_base=p_base, c_base=c_base, p_attack=p_attack, c_attack=c_attack),3)}')

def make_als_plots(cov_adjust_strength=3.0, pairs='v3'):
    als = alscore.ALScore()
    als.set_param('cov_adjust_strength', cov_adjust_strength)
    if pairs == 'v1':
        Catk_Cbase_pairs = [(1, 1), (0.01, 0.01), (0.7, 1.0), (0.01, 0.05)]
        fig, axs = plt.subplots(2, 2, figsize=(7, 6))
    elif pairs == 'v2':
        Catk_Cbase_pairs = [(1, 1), (0.1, 0.1), (0.01, 0.01), (0.001, 0.001)]
        fig, axs = plt.subplots(2, 2, figsize=(7, 6))
    elif pairs == 'v3':
        Catk_Cbase_pairs = [(1, 1), (0.1, 0.1), (0.01, 0.01), (0.001, 0.001), (0.075, 0.1), (0.05, 0.01)]
        fig, axs = plt.subplots(3, 2, figsize=(6, 8))
    Pbase_values = [0.01, 0.1, 0.4, 0.7, 0.9]
    Patk = np.arange(0, 1.01, 0.01)
    
    axs = axs.flatten()
    
    for i, (Catk, Cbase) in enumerate(Catk_Cbase_pairs):
        for Pbase in Pbase_values:
            ALC = [als.alscore(p_base=Pbase, c_base=Cbase, p_attack=p, c_attack=Catk) for p in Patk]
            axs[i].plot(Patk, ALC, label=f'Pbase={Pbase}')
        
        axs[i].text(0.05, 0.95, f'Catk = {Catk}, Cbase = {Cbase}\nalpha = {cov_adjust_strength}\nCmin = 0.0001', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
        axs[i].set_xlim(0, 1)
        axs[i].set_ylim(-0.5, 1)
        
        # Remove x-axis labels and ticks for the upper two subplots
        if i < len(Catk_Cbase_pairs) - 2:
            axs[i].set_xlabel('')
            #axs[i].set_xticklabels([])
        
        # Remove y-axis labels and ticks for the right subplots
        if i % 2 == 1:
            axs[i].set_ylabel('')
            axs[i].set_yticklabels([])
        
        if i % 2 == 0:
            axs[i].set_ylabel('ALC')
        
        if i >= len(Catk_Cbase_pairs) - 2:
            axs[i].set_xlabel('Patk')
        
        axs[i].legend(fontsize='small', loc='lower right')
        axs[i].grid(True)
    
    plt.tight_layout()
    
    # Save the plot in both PNG and PDF formats
    plt.savefig(f'als_plots/als_plot_{cov_adjust_strength}_{pairs}.png')
    plt.savefig(f'als_plots/als_plot_{cov_adjust_strength}_{pairs}.pdf')

als = alscore.ALScore()
do_als_test(als, p_base=0.5, c_base=1.0, increase=0.2, c_attack=1.0)
do_als_test(als, p_base=0.2, c_base=1.0, increase=0.8, c_attack=1.0)
do_als_test(als, p_base=0.999, c_base=1.0, increase=0.9, c_attack=1.0)
do_als_test(als, p_base=0.5, c_base=0.1, increase=0.2, c_attack=0.1)
do_als_test(als, p_base=0.2, c_base=0.1, increase=0.8, c_attack=0.1)
do_als_test(als, p_base=0.5, c_base=0.01, increase=0.2, c_attack=0.01)
do_als_test(als, p_base=0.2, c_base=0.01, increase=0.8, c_attack=0.01)
do_als_test(als, p_base=0.5, c_base=0.001, increase=0.2, c_attack=0.001)
do_als_test(als, p_base=0.2, c_base=0.001, increase=0.8, c_attack=0.001)
do_als_test(als, p_base=0.5, c_base=0.0001, increase=0.2, c_attack=0.0001)
do_als_test(als, p_base=0.2, c_base=0.0001, increase=0.8, c_attack=0.0001)
do_als_test(als, p_base=1.0, c_base=0.00001, increase=0, c_attack=0.00001)
plot_prec_cov_for_equal_pcc('prec_cov_for_equal_pcc')
make_als_plots(pairs='v3')
for cov_adjust_strength in [1.0, 2.0, 3.0, 4.0]:
    make_als_plots(cov_adjust_strength=cov_adjust_strength, pairs='v1')
    make_als_plots(cov_adjust_strength=cov_adjust_strength, pairs='v2')
