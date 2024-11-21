import numpy as np
import os
import matplotlib.pyplot as plt
import alscore
import math
from alc_plot_basic import plot_basic_alc
from cov_plot import plot_coverage_alc
from cov_mismatch import plot_alscore_scatter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pprint
pp = pprint.PrettyPrinter(indent=4)

als = alscore.ALScore()

plots_path = os.path.join('alc_plots')
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

def savefigs(plt, name):
    for suffix in ['.png', '.pdf']:
        path_name = name + suffix
        out_path = os.path.join(plots_path, path_name)
        plt.savefig(out_path)

def run_test(als, pcc_base, pcc_atk):
    alc = als.alscore(pcc_base=pcc_base, pcc_attack=pcc_atk)
    new_pcc_atk = als.pccatk_from_pccbase_alc(pcc_base, alc)
    if not math.isclose(pcc_atk, new_pcc_atk, rel_tol=1e-4):
        print(f'pcc_base: {pcc_base}, pcc_atk: {pcc_atk}, alc: {alc}, new_pcc_atk: {new_pcc_atk}')

# Run some tests of als.pccatk_from_pccbase_alc(pcc_base, alc):
run_test(als, pcc_base=0.5, pcc_atk=0.5)
run_test(als, pcc_base=0.5, pcc_atk=0.8)
run_test(als, pcc_base=0.8, pcc_atk=0.5)
run_test(als, pcc_base=0.99, pcc_atk=0.995)
run_test(als, pcc_base=0.995, pcc_atk=0.99)
run_test(als, pcc_base=0.99995, pcc_atk=0.9999)
run_test(als, pcc_base=0.01, pcc_atk=0.9999)
run_test(als, pcc_base=0, pcc_atk=0.9999)
print(f'for PCCbase = 0.5 and ALC = 0.5, PCCatk = {als.pccatk_from_pccbase_alc(0.5, 0.5)}')
print(f'for PCCbase = 0.05 and ALC = 0.5, PCCatk = {als.pccatk_from_pccbase_alc(0.05, 0.5)}')
print(f'for PCCbase = 0.95 and ALC = 0.5, PCCatk = {als.pccatk_from_pccbase_alc(0.95, 0.5)}')

plt = plot_alscore_scatter(als)
savefigs(plt, 'cov_mismatch')
plt = plot_coverage_alc(als)
savefigs(plt, 'alc_cov')
plt = plot_basic_alc(als)
savefigs(plt, 'alc_basic')