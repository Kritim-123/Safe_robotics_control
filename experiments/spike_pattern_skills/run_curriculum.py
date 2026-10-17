"""Record a progressive terrain benchmark; assessment gates never train weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import uuid

from hybrid.arena import build_model
from .associations import AssociationMemory
from .curriculum import STAGES, curriculum
from .encoder import RandomLIFEncoder
from .execute_associations import DEFAULT_ASSOCIATION, execute_case
from .learn_associations import ENCODER_PATH, write_json
from .record import ROOT


def assessment(records, modes, stages):
    groups, gates = [], {}
    for mode in modes:
        unlocked = True
        previous = -1
        for stage in stages:
            unlocked = unlocked and stage == previous+1
            previous = stage
            group = [r for r in records if r['stage'] == stage and r['mode'] == mode]
            development = [r for r in group if r['split'] == 'development']
            checks = [r for r in group if r['split'] != 'development']
            complete = len(group) == 6 and len(development) == len(checks) == 3
            passed = complete and all(sum(r['traversal_success'] for r in rows)/len(rows) >= .8
                and not any(r['collision'] or r['fall'] or r.get('error') for r in rows)
                for rows in (development, checks))
            groups.append(dict(mode=mode, stage=stage, family=STAGES[stage], episodes=len(group),
                traversals=sum(r['traversal_success'] for r in group),
                primitive_completions=sum(r.get('primitive_completed', False) for r in group),
                contacts=sum(r['collision'] for r in group), falls=sum(r['fall'] for r in group),
                skills=dict(Counter(r.get('skill') for r in group)),
                reasons=dict(Counter(r['reason'] for r in group)),
                eligible_for_progression=unlocked, stage_passed=bool(passed),
                progression_status=('passed' if passed else 'needs_work') if unlocked else 'locked',
                complete_six_variant_assessment=complete))
            unlocked = unlocked and passed
        gates[mode] = [g['stage'] for g in groups if g['mode'] == mode and g['progression_status'] == 'passed']
    return dict(episodes=len(records), groups=groups, passed_stages=gates,
                note='Diagnostic sweep may run locked stages; no model or skill updates occur.')


def run(output=None, seed=20261017, stages=None, variants=None, modes=None,
        viewer=False, video=False, compile_only=False, progressive=False):
    stages = list(range(10)) if stages is None else stages
    variants = list(range(6)) if variants is None else variants
    modes = modes or ['association', 'direct_run']
    if (not stages or not variants or len(set(stages)) != len(stages) or stages != sorted(stages)
            or len(set(variants)) != len(variants) or any(s not in range(10) for s in stages)
            or any(v not in range(6) for v in variants)
            or len(set(modes)) != len(modes) or any(m not in ('association', 'direct_run') for m in modes)):
        raise ValueError('Require ordered unique stages, unique variants and supported modes')
    if progressive and (stages != list(range(max(stages)+1)) or variants != list(range(6))):
        raise ValueError('Progressive runs must start at stage zero and assess all six variants in order')
    folder = Path(output) if output else ROOT/'output/spike_pattern_skills/curriculum'/uuid.uuid4().hex
    if not folder.is_absolute():
        folder = ROOT/folder
    folder.mkdir(parents=True, exist_ok=False)
    cases = [c for c in curriculum(seed) if c.stage in stages and c.variant in variants]
    encoder = RandomLIFEncoder.load(ENCODER_PATH)
    memory = AssociationMemory.load(DEFAULT_ASSOCIATION, encoder.checksum)
    manifest = dict(seed=seed, stages=stages, variants=variants, modes=modes,
        duration=8., timeout=18., encoder_checksum=encoder.checksum, association_checksum=memory.checksum,
        progressive=progressive,
        geometry_sha256=hashlib.sha256(json.dumps([c.scenario.to_dict() for c in cases],
            sort_keys=True).encode()).hexdigest(), scenarios=len(cases), training=False,
        source_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'hybrid/terrain.py', ROOT/'hybrid/arena.py', ROOT/'skills/dynamic.py',
                      ROOT/'experiments/spike_pattern_skills/curriculum.py',
                      ROOT/'experiments/spike_pattern_skills/execute_associations.py', Path(__file__).resolve()]})
    write_json(folder/'manifest.json', manifest)
    config = folder/'scenarios'; config.mkdir()
    for case in cases:
        write_json(config/(case.scenario.name+'.json'), case.scenario.to_dict())
        if compile_only:
            model = build_model(case.scenario)
            print(case.scenario.name, 'compiled', model.ngeom, flush=True)
    if compile_only:
        return manifest
    records = []
    allowed = {mode: True for mode in modes}
    with (folder/'results.jsonl').open('x', encoding='utf-8') as stream:
        for case in cases:
            if progressive and not any(allowed.values()):
                break
            for mode in modes:
                if progressive and not allowed[mode]:
                    continue
                episode = folder/'episodes'/(case.scenario.name+'-'+mode)
                base = dict(context_id=case.scenario.name, stage=case.stage, family=case.family,
                            variant=case.variant, split=case.split, scene_seed=case.scene_seed, mode=mode)
                try:
                    result = execute_case(case, memory, encoder, episode, viewer=viewer, video=video,
                                          duration=8., timeout=18., selection=mode)
                    summary, measured = result['summary'], result['outcome']
                    row = dict(base, skill=summary['skill'], reason=summary['reason'],
                        traversal_success=measured['task_fulfilled'], primitive_completed=summary['success'],
                        collision=summary['collision'], fall=summary['fall'],
                        final_position=measured['final_position'], progress=measured['body_forward_progress'],
                        max_torque=summary['max_torque'], max_torque_limit_ratio=summary['max_torque_limit_ratio'],
                        fallback=result['decision'].get('supervision', {}).get('fallback', True),
                        result_path=str((episode/'result.json').relative_to(folder)).replace('\\', '/'))
                except Exception as error:
                    row = dict(base, traversal_success=False, primitive_completed=False,
                               collision=False, fall=False, reason='runner error', error=str(error))
                records.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                write_json(folder/'assessment.json', assessment(records, modes, stages))
                print(case.scenario.name, mode, row['reason'], 'traversed='+str(row['traversal_success']), flush=True)
                if row['reason'] == 'viewer closed':
                    return assessment(records, modes, stages)
            if progressive and case.variant == 5:
                current = assessment(records, modes, stages)
                for mode in modes:
                    stage_result = next(g for g in current['groups'] if g['mode'] == mode and g['stage'] == case.stage)
                    allowed[mode] = stage_result['progression_status'] == 'passed'
    report = assessment(records, modes, stages)
    report['errors'] = sum('error' in r for r in records)
    write_json(folder/'assessment.json', report)
    if report['errors']:
        raise RuntimeError('Curriculum had runner errors; partial evidence retained')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--stages', type=int, nargs='+')
    parser.add_argument('--variants', type=int, nargs='+')
    parser.add_argument('--modes', choices=('association', 'direct_run'), nargs='+')
    for flag in ('viewer', 'video', 'compile-only', 'progressive'):
        parser.add_argument('--'+flag, action='store_true')
    run(**vars(parser.parse_args()))
