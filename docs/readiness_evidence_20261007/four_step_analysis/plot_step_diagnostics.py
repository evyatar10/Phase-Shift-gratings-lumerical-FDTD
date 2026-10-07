"""Plot measured diagnostics only; requires the preserved report and both raw probes."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
base = Path(__file__).resolve().parent.parent
report = json.loads((base/'s2_corr30_four_steps_measured.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout='constrained')
for metric, label, color in [('peak3_tout', 'Port T_out', '#137f75'), ('peak3_modal', 'Modal |S21|^2', '#356bb3')]:
    steps = np.array([.25, .5, 1., 2.])
    values = [report['observables'][metric]['steps'][str(h)]['scalar_with_new_center']['central_derivative']*1e5 for h in steps]
    axes[0].plot(steps, values, 'o-', color=color, label=label, lw=1.7, ms=5)
axes[0].set(xscale='log', xlabel='Forward finite-difference half-step (nm)', ylabel='Peak transmission derivative (10^-5 / nm)', title='Forward sensitivity depends on step')
for quad, label, color in [('re', 'Re component', '#8156a3'), ('im', 'Im component', '#d27d29')]:
    data = json.loads((base/f's2_deps_{quad}_report_original.json').read_text())
    steps = np.array([row['step_nm'] for row in data['raw_steps']])
    values = np.array([row['raw_contraction'] for row in data['raw_steps']])
    axes[1].plot(steps, 100*(values/values[0]-1), 'o-', label=label, color=color, lw=1.7, ms=5)
axes[1].set(xscale='log', xlabel='Permittivity derivative half-step (nm)', ylabel='Change from own 1 nm result (%)', title='Stored-field gradient components')
for index, ax in enumerate(axes):
    ticks = [.25, .5, 1, 2] if index == 0 else [.25, .5, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(x) for x in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=9)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
fig.suptitle('S2 corr30 diagnostics — global calibration remains FAILED', fontsize=13)
fig.savefig(base/'s2_corr30_step_diagnostics.png', dpi=160)
fig.savefig(base/'s2_corr30_step_diagnostics.svg')
