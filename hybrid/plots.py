"""Make shareable experiment figures from saved records, without rerunning physics."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from hybrid.snn import DEFAULT_MODEL, SpikingSelector
from skills.avoidance import SKILL_NAMES

COLORS = {'baseline': '#b24b39', 'rules': '#b18a21', 'snn': '#176d9c',
          'BASELINE': '#176d9c', 'SKILL': '#cf623a', 'RECOVER': '#7755aa'}


def plot_run(directory, output=None):
    directory = Path(directory)
    output = Path(output) if output else directory/'trajectory.png'
    summary = json.loads((directory/'summary.json').read_text())
    spec = json.loads((directory/'scenario.json').read_text())
    events = json.loads((directory/'events.json').read_text())
    with (directory/'trajectory.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    t, x, y = (np.array([float(row[key]) for row in rows]) for key in ('time', 'x', 'y'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained', width_ratios=[1.25, 1])
    ax = axes[0]
    x0, x1, y0, y1 = spec['bounds']
    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='#64748b', linewidth=2))
    for obstacle in spec['obstacles']:
        start = np.array(obstacle['center'])
        end = start + np.array(obstacle['velocity'])*min(summary['elapsed'], obstacle['stop_after'])
        ax.add_patch(Circle(start, obstacle['radius'], color='#b24b39', alpha=.28))
        ax.add_patch(Circle(end, obstacle['radius'], color='#b24b39', alpha=.6))
        if not np.allclose(start, end):
            ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='#b24b39'))
    for state in ('BASELINE', 'SKILL', 'RECOVER'):
        selected = np.array([r['state'] == state for r in rows])
        ax.plot(np.where(selected, x, np.nan), np.where(selected, y, np.nan), color=COLORS[state], label=state, lw=2)
    ax.scatter(*spec['start'], marker='o', color='#239b60', s=55, label='start', zorder=4)
    ax.scatter(*spec['goal'], marker='*', color='#daa520', s=150, label='goal', zorder=4)
    ax.set(xlabel='World x (m)', ylabel='World y (m)', aspect='equal',
           title=f"{summary['scenario']} / {summary['mode']}: {summary['reason']}")
    ax.legend(fontsize=8, loc='lower left', ncol=2)
    ax.grid(alpha=.15)
    ax = axes[1]
    ax.plot(t, [float(r['distance']) for r in rows], color='#176d9c', label='goal distance')
    ax.plot(t, [float(r['clearance']) for r in rows], color='#b24b39', label='footprint clearance proxy')
    if rows and 'observed_clearance' in rows[0]:
        measured = np.array([float(r['observed_clearance']) for r in rows])
        actual = np.array([float(r['clearance']) for r in rows])
        if np.max(np.abs(measured-actual)) > .005:
            ax.plot(t, measured, color='#b18a21', alpha=.7, lw=.8, label='observed clearance')
    ax.axhline(.12, color='#64748b', linestyle=':', label='arrival tolerance')
    for event in events:
        if event['target'] in ('SKILL', 'RECOVER', 'BASELINE'):
            ax.axvline(event['time'], color=COLORS[event['target']], alpha=.4, lw=1)
            if event['target'] == 'SKILL':
                ax.text(event['time'] + .5, 2.6, event['skill'], rotation=90,
                        va='top', fontsize=8, color=COLORS[event['target']])
        elif event['target'] == 'FAILED' and 'observed_clearance' in event:
            ax.scatter(event['time'], event['observed_clearance'], marker='X', s=55,
                       color='#b24b39', label='observed clearance at stop', zorder=5)
    ax.set(xlabel='Simulation time (s)', ylabel='Distance (m)', title='Handoffs and clearance')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=.15)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    selections = [e for e in events if e['target'] == 'SKILL' and e.get('spikes', 0) > 0]
    if selections:
        event = selections[0]
        selector = SpikingSelector()
        # Refuse to label a different model's raster as this recorded inference.
        import hashlib
        if hashlib.sha256(selector.path.read_bytes()).hexdigest() == summary['model_sha256']:
            scores, diagnostics = selector.predict(event['features'], trace=True)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained', width_ratios=[1.7, 1])
            times, neurons = np.nonzero(diagnostics['raster'])
            axes[0].scatter(times+1, neurons, marker='|', s=10, color='#176d9c')
            axes[0].axhline(63.5, color='#b24b39', linestyle='--', lw=1)
            axes[0].set(xlabel='Internal SNN step (not elapsed sensor time)', ylabel='Hidden neuron',
                        title=f"Binary spikes at t={event['time']:.1f}s", xlim=(.5,16.5), ylim=(-1,96))
            axes[1].barh(SKILL_NAMES, scores, color=['#176d9c' if a else '#ccc' for a in event['allowed']])
            axes[1].set(xlabel='Readout score (not probability)', title=f"Selected: {event['skill']}")
            fig.savefig(output.with_name(output.stem+'-spikes.png'), dpi=160)
            plt.close(fig)
    return output


def plot_evaluation(directory):
    directory = Path(directory)
    report = json.loads((directory/'evaluation.json').read_text())
    names, modes = report['scenarios'], report['modes']
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout='constrained')
    for i, mode in enumerate(modes):
        groups = [next(g for g in report['aggregates'] if g['scenario']==name and g['mode']==mode) for name in names]
        x = np.arange(len(names)) + (i-(len(modes)-1)/2)*.25
        axes[0].bar(x, [g['successes']/g['trials']*100 for g in groups], width=.24, label=mode, color=COLORS[mode])
        axes[1].bar(x, [g['collisions']/g['trials']*100 for g in groups], width=.24, label=mode, color=COLORS[mode])
    for ax, title in zip(axes, ['Goal success (%)', 'Physical contact failures (%)']):
        ax.set(xticks=np.arange(len(names)), xticklabels=[n.replace('_',' ') for n in names], ylim=(0,110), ylabel=title)
        ax.grid(axis='y', alpha=.2)
        ax.legend(ncol=len(modes), loc='upper right')
    sensing = report.get('sensing', {})
    imperfect = any(sensing.get(key, 0) for key in ('position_noise', 'velocity_noise', 'sensor_delay'))
    subtitle = 'noisy/delayed simulator observations' if imperfect else 'ground-truth perception'
    axes[0].set_title(f"Paired MuJoCo trials: {len(report['seeds'])} seeds per scenario and mode; {subtitle}")
    output = directory/'comparison.png'
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_training():
    report = json.loads((DEFAULT_MODEL.parent/'training.json').read_text())
    curve = report['epochs']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot([r['epoch'] for r in curve], [r['validation_accuracy']*100 for r in curve], color='#176d9c')
    axes[0].set(xlabel='Epoch', ylabel='Masked validation agreement (%)', title='Geometric teacher imitation')
    matrix = np.asarray(report['confusion_matrix'])
    axes[1].imshow(matrix, cmap='Blues')
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, str(matrix[i,j]), ha='center', va='center', color='white' if matrix[i,j]>450 else 'black')
    axes[1].set(xticks=range(4), yticks=range(4), xticklabels=SKILL_NAMES, yticklabels=SKILL_NAMES,
                xlabel='SNN prediction', ylabel='Teacher label', title='Independent synthetic test set (n=2,000)')
    axes[1].tick_params(axis='x', labelrotation=30)
    output = Path('output/hybrid/training.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--training', action='store_true')
    args = parser.parse_args()
    if args.run:
        print(plot_run(args.run))
    if args.evaluation:
        print(plot_evaluation(args.evaluation))
    if args.training:
        print(plot_training())


if __name__ == '__main__':
    main()
