"""Replay saved SNN decisions into inspectable plots; never simulate fake activity."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hybrid.snn import DEFAULT_MODEL, SpikingSelector
from hybrid.perception import FEATURE_NAMES
from skills.avoidance import SKILL_NAMES


def build_report(directory, model=DEFAULT_MODEL):
    directory, model = Path(directory), Path(model)
    summary = json.loads((directory/'summary.json').read_text())
    events = json.loads((directory/'events.json').read_text())
    if summary['mode'] != 'snn':
        raise ValueError('Brain replay requires an SNN run')
    if summary['model_sha256'] != hashlib.sha256(model.read_bytes()).hexdigest():
        raise ValueError('Model hash differs from the recorded run; cannot replay faithfully')
    selector = SpikingSelector(model)
    out = directory/'brain'
    out.mkdir(exist_ok=True)
    with (directory/'trajectory.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    times = np.array([float(r['time']) for r in rows])
    has_commands = bool(rows and 'command_vx' in rows[0])
    fig, axes = plt.subplots(4 if has_commands else 3, 1, figsize=(12, 11 if has_commands else 9), sharex=True, layout='constrained')
    for key in ('distance', 'speed'):
        index = 0 if key == 'distance' else 1
        axes[index].plot(times, [float(r[key]) for r in rows], label=key)
    axes[0].set_ylabel('Distance to goal (m)')
    axes[1].set_ylabel('Measured speed (m/s)')
    axes[2].plot(times, [float(r['torque']) for r in rows], color='#a54a16')
    axes[2].set_ylabel('Max |joint torque| (Nm)')
    if has_commands:
        for key in ('command_vx', 'command_vy'):
            axes[3].plot(times, [float(r[key]) for r in rows], label=key)
        axes[3].set_ylabel('Body translation command (m/s)')
        axes[3].legend()
    axes[-1].set_xlabel('Simulation time (s) — PID continues between SNN decisions')
    for event in events:
        if 'features' in event:
            for ax in axes:
                ax.axvline(event['time'], color='#555', alpha=.4)
            axes[0].annotate(event['skill'], (event['time'], 0), xytext=(4, 12),
                             textcoords='offset points', rotation=90)
    fig.suptitle('Actual robot signals: motion, pauses and motor effort')
    fig.savefig(out/'episode.png', dpi=140)
    plt.close(fig)
    sections = []
    traces = {}
    for index, event in enumerate(e for e in events if 'features' in e):
        scores, trace = selector.predict(event['features'], trace=True)
        np.testing.assert_allclose(scores, event['scores'], atol=2e-5, rtol=2e-5)
        traces[f'decision_{index+1}'] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in trace.items()}
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
        ax = axes[0, 0]
        labels = [f'{n.replace("_", " ")} = {v:.3f}' for n, v in zip(FEATURE_NAMES, event['features'])]
        ax.barh(labels, trace['normalized_inputs'], color='#247a91')
        ax.invert_yaxis()
        ax.set_xlabel('Normalized input current encoding (clipped to ±5)')
        ax.set_title('1. Observations → constant input to layer 1')
        ax = axes[0, 1]
        step, neuron = np.nonzero(trace['raster'])
        ax.scatter(step+1, neuron, marker='|', s=22, color='#247a91')
        ax.axhline(63.5, color='#a54a16', linestyle='--')
        ax.set(xlabel='Internal SNN step (not seconds)', ylabel='Neuron: L1 0–63; L2 64–95',
               xlim=(.5, selector.steps+.5), ylim=(-1, 96), title='2. Binary spikes across both hidden layers')
        ax = axes[1, 0]
        # Deterministic representative active layer-2 neuron; not a causal explanation.
        neuron = 64 + int(np.argmax(trace['raster'][:, 64:].sum(axis=0)))
        x = np.arange(1, selector.steps+1)
        ax.plot(x, trace['membrane_before_reset'][:, neuron], 'o-', label='Before reset')
        ax.plot(x, trace['membrane_after_reset'][:, neuron], '.--', label='After subtracting spike')
        ax.axhline(1, color='#a54a16', linestyle=':', label='Firing threshold = 1')
        ax.set(xlabel='Internal SNN step', ylabel='Model membrane value (dimensionless)',
               title=f'3. Layer-2 neuron {neuron-64}: leak, integrate, fire, reset')
        ax.legend(fontsize=9)
        ax = axes[1, 1]
        for i, name in enumerate(SKILL_NAMES):
            allowed = event['allowed'][i]
            ax.plot(x, trace['readouts'][:, i], label=f'{name}: {scores[i]:.2f}' + (' [MASKED]' if not allowed else ''),
                    linestyle='-' if allowed else '--')
        ax.set(xlabel='Internal SNN step', ylabel='Running mean-spike readout (not probability)',
               title='4. Final scores → safety mask → selected skill')
        ax.legend(fontsize=9)
        title = f'Decision {index+1} at {event["time"]:.1f}s: {event["skill"]} — {trace["spikes"]} spikes'
        fig.suptitle(title, fontsize=16)
        filename = f'decision-{index+1}.png'
        fig.savefig(out/filename, dpi=140)
        plt.close(fig)
        sections.append(f'<h2>{html.escape(title)}</h2><p>{html.escape(event["reason"])}</p><img src="{filename}" alt="Input values, spike raster, membrane trace and skill scores">')
    (out/'traces.json').write_text(json.dumps(traces, allow_nan=False), encoding='utf-8')
    (out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Go2 SNN decision replay</title>
<style>body{font:17px system-ui;max-width:1300px;margin:30px auto;padding:0 20px;color:#152b37;background:#fff}img{width:100%;height:auto}p{max-width:950px;line-height:1.6}</style>
<h1>Inside the Go2 skill selector</h1><p>Saved observations → 12 inputs → 64 spiking neurons → 32 spiking neurons → 4 skill scores → geometric safety mask → waypoint skill → gait → joint PID torques.</p>
<p>These are deterministic replays using the exact recorded model hash. Weights are frozen: the robot is not learning during this run. Each decision starts with zero membrane values and lasts 16 internal steps. Spikes are binary model events; membrane values are dimensionless, not biological voltages. The SNN selects a skill; it does not directly generate motor torques.</p>
<img src="episode.png" alt="Recorded goal distance, robot speed and joint torque over simulation time">
''' + ''.join(sections) + '<p>No SNN decisions between the marked events. PID control continues during walking and waiting. A representative neuron illustrates dynamics, not proof of what an individual neuron means.</p></html>', encoding='utf-8')
    return out/'index.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('output/hybrid/latest'))
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(build_report(args.run, args.model))
