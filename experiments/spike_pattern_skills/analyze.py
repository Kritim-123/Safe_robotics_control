"""Compare active-ID, count-vector and time-raster representations honestly."""
from collections import defaultdict
from itertools import combinations

import numpy as np


def analyze(records):
    if not records:
        raise ValueError('No records to analyze')
    groups = {name: defaultdict(list) for name in ('active_ids', 'spike_counts', 'raster')}
    for index, record in enumerate(records):
        r = record['recording']
        raster = np.asarray(r['output_raster'])
        counts = np.asarray(r['spike_counts'])
        if (raster.ndim != 2 or counts.shape != (raster.shape[1],)
                or not np.isin(raster, [0, 1]).all() or not np.array_equal(raster.sum(0), counts)
                or np.flatnonzero(counts).tolist() != r['active_neuron_ids']):
            raise ValueError('Inconsistent spike raster, counts or active IDs')
        groups['active_ids'][tuple(r['active_neuron_ids'])].append(index)
        groups['spike_counts'][tuple(r['spike_counts'])].append(index)
        groups['raster'][tuple(raster.ravel().tolist())].append(index)
    def family(index):
        return records[index]['observation']['metadata']['study_family']
    def context(index):
        return records[index]['observation']['context_id']
    metrics = {}
    for name, mapping in groups.items():
        collided = [indices for indices in mapping.values() if len(indices) > 1]
        metrics[name] = dict(unique_patterns=len(mapping), shared_groups=len(collided),
            shared_pairs=sum(len(g)*(len(g)-1)//2 for g in collided),
            cross_family_shared_pairs=sum(family(a) != family(b) for g in collided for a, b in combinations(g, 2)),
            shared_contexts=[[context(i) for i in group] for group in collided])
    collisions = [pair for group in groups['active_ids'].values() for pair in combinations(group, 2)]
    resolved = sum(records[a]['recording']['spike_counts'] != records[b]['recording']['spike_counts']
                   for a, b in collisions)
    counts = np.array([r['recording']['spike_counts'] for r in records])
    families = sorted({family(i) for i in range(len(records))})
    example = next(((a, b) for a, b in collisions if family(a) != family(b) and
                    records[a]['recording']['spike_counts'] != records[b]['recording']['spike_counts']), None)
    return dict(observations=len(records), representations=metrics,
        active_id_shared_pairs_resolved_by_counts=resolved,
        active_id_shared_pairs_still_ambiguous=len(collisions)-resolved,
        silent_observations=int(np.sum(counts.sum(1) == 0)),
        total_spikes=int(counts.sum()), mean_spikes_per_observation=float(counts.sum(1).mean()),
        neurons=[dict(neuron=i, active_in_situations=int(np.sum(counts[:, i] > 0)),
            mean_count=float(counts[:, i].mean()), min_count=int(counts[:, i].min()),
            max_count=int(counts[:, i].max()), total_count=int(counts[:, i].sum())) for i in range(counts.shape[1])],
        family_summary={f: dict(situations=sum(family(i) == f for i in range(len(records))),
            mean_total_spikes=float(np.mean([r['recording']['spike_counts'] for i, r in enumerate(records)
                                             if family(i) == f], axis=0).sum())) for f in families},
        example_cross_family_id_collision=([dict(context=context(i), family=family(i),
            active_ids=records[i]['recording']['active_neuron_ids'],
            spike_counts=records[i]['recording']['spike_counts']) for i in example] if example else []),
        interpretation='Representation collisions are not skill-selection errors: no task labels or decoder exist.')
