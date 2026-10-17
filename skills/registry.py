"""Discover callable skills without confusing them with learned SNN classes.

Imported command options require their saved validation. Explicit experimental
classical skills carry their measured limitations and remain outside the SNN.
The legacy four-action and experimental three-action SNN schemas are unchanged.
"""
import argparse
import json

from skills.pretrained_velocity import COMMANDS, POLICY_ROOT


def list_skills(include_unavailable=False):
    items = []
    for group, names, entry in (
            ('standalone', ('walk', 'navigate'), 'skills.walk / skills.navigate'),
            ('avoidance', ('bypass_left', 'bypass_right', 'wait', 'back_off'), 'skills.avoidance'),
            ('dynamic', ('wait', 'run', 'jump'), 'skills.dynamic')):
        for name in names:
            items.append(dict(identifier=f'{group}/{name}', implementation='classical',
                              available=True, selector_class=(name if group in ('avoidance', 'dynamic') else None),
                              selector_schema=('legacy_4' if group == 'avoidance' else
                                               ('skill_lif_3' if group == 'dynamic' else None)),
                              entry_point=entry, validation='See existing project milestone reports'))
    for name in ('yield_hold', 'retreat_hold'):
        items.append(dict(identifier='temporal/'+name, implementation='classical composite', available=False,
                          selector_class=None, selector_schema=None, entry_point='skills.temporal',
                          validation='Experimental: documented timeouts behind persistent blockers'))
    items.append(dict(identifier='standalone/low_bar_jump', implementation='classical quintic jump',
        available=True, experimental=True, snn_eligible=False, selector_class=None, selector_schema=None,
        entry_point='skills.obstacle_jump.execute_low_bar_jump',
        validation='docs/agents/obstacle_jump_validation.md; nominal crossing passed; nearby +X 14/16, landing contacts remain'))
    validation_path = POLICY_ROOT/'validation.json'
    validation = json.loads(validation_path.read_text(encoding='utf-8')) if validation_path.exists() else {}
    validated = validation.get('skills', {})
    for policy in ('robot_lab', 'himloco'):
        manifest = json.loads((POLICY_ROOT/policy/'manifest.json').read_text(encoding='utf-8'))
        for name, command in COMMANDS.items():
            identifier = f'pretrained/{policy}/{name}'
            evidence = validated.get(identifier, {})
            matched = (evidence.get('source_policy_sha256') == manifest['artifacts']['source_policy.pt']
                       and evidence.get('export_sha256') == manifest['artifacts']['model.npz'])
            items.append(dict(identifier=identifier, implementation='pretrained velocity command option',
                              learned_policy=policy, available=bool(evidence.get('enabled', False) and matched),
                              command=list(command), selector_class=None, selector_schema=None,
                              snn_eligible=False, entry_point='skills.pretrained_velocity.execute_pretrained_skill',
                              validation=evidence or 'Not evaluated'))
    items.append(dict(identifier='pretrained/atomic/gap_jump', implementation='pretrained motor policy',
                      available=False, learned_policy='hando_w30', snn_eligible=False,
                      selector_class=None, selector_schema=None, entry_point='skills.pretrained_jump',
                      validation='Recorded 0/7 crossings; excluded from selection'))
    return [item for item in items if include_unavailable or item['available']]


def get_skill(identifier):
    for item in list_skills(include_unavailable=True):
        if item['identifier'] == identifier:
            return item
    raise ValueError(f'Unknown skill identifier: {identifier}')


def run_skill(identifier, **kwargs):
    """Dispatch a validated external option; caller may supply a Scenario.

    This wraps a complete standalone episode, not a persistent HybridSystem
    handoff. The adapter class exposes target generation for future integration.
    """
    item = get_skill(identifier)
    if not item['available']:
        raise ValueError(f'Skill unavailable after validation: {identifier}')
    if identifier == 'standalone/low_bar_jump':
        from skills.obstacle_jump import execute_low_bar_jump
        return execute_low_bar_jump(**kwargs)
    parts = identifier.split('/')
    if len(parts) != 3 or parts[0] != 'pretrained':
        raise ValueError('Use the existing entry point for classical skills')
    from skills.pretrained_velocity import execute_pretrained_skill
    return execute_pretrained_skill(parts[1], parts[2], **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Also show unavailable experiments')
    args = parser.parse_args()
    print(json.dumps(list_skills(include_unavailable=args.all), indent=2))
