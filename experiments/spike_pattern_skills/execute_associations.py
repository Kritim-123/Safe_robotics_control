"""Execute frozen count associations and flush observations/decisions live to disk."""
import argparse
from collections import Counter
import json
from pathlib import Path
import uuid

import numpy as np

from skills.dynamic import execute_dynamic_skill
from .associations import AssociationMemory, outcome, supervise
from .encoder import RandomLIFEncoder
from .learn_associations import ENCODER_PATH, save_trajectory, write_json
from .observations import observation_from_simulation
from .record import ROOT
from .situations import situations

DEFAULT_ASSOCIATION = ROOT/'docs/agents/pattern_association_v1/association.json'


def measure_case_outcome(case, history, summary):
    measured = outcome(summary, case.jump_requested)
    if getattr(case, 'traversal_x', None) is not None:
        terminal = history[-1]
        traversed = (summary['success'] and not summary['collision'] and not summary['fall']
            and summary['max_torque_limit_ratio'] <= 1+1e-9
            and terminal['x'] >= case.traversal_x and abs(terminal['y']) <= .35)
        measured.update(task='terrain_traversal',
            forward_progress_achieved=measured['task_fulfilled'], task_fulfilled=bool(traversed),
            utility=measured['body_forward_progress'] if traversed else None,
            final_position=[terminal['x'], terminal['y']], traversal_x=case.traversal_x)
    return measured

def execute_case(case, memory, encoder, folder, *, viewer=False, video=False,
                 timeout=12., interrupt_after=None, duration=4., selection='association',
                 jump_profile='legacy', jump_config=None):
    if selection not in ('association', 'direct_run'):
        raise ValueError('Unknown selection mode')
    if memory.encoder_checksum != encoder.checksum:
        raise ValueError('Association requires a different encoder')
    observe = getattr(encoder, 'observation_adapter', observation_from_simulation)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    decision, latest = {}, {}
    next_sample = 0.
    last_overlay = None
    renderer = movie = None
    if video:
        import imageio.v2 as imageio
        movie = imageio.get_writer(str(folder/'recording.mp4'), fps=10)

    with (folder/'telemetry.jsonl').open('x', encoding='utf-8') as telemetry, \
            (folder/'events.jsonl').open('x', encoding='utf-8') as events:
        def event(kind, **values):
            events.write(json.dumps(dict(event=kind, **values), allow_nan=False)+'\n')
            events.flush()

        event('start', context_id=case.scenario.name, encoder_checksum=encoder.checksum,
              association_checksum=memory.checksum, jump_requested=case.jump_requested,
              selection_mode=selection, duration=duration, timeout=timeout,
              jump_profile=jump_profile,
              scenario=case.scenario.to_dict(), reset_policy='fresh episode; no reset during skill')

        def select(observation, eligibility):
            item = observe(case.scenario, observation, case.jump_requested,
                observation_id=case.scenario.name+'-decision', context_id=case.scenario.name)
            firing = encoder.encode(item['features'])
            proposal = memory.decode(firing['spike_counts'])
            chosen = supervise(proposal, eligibility)
            if selection == 'direct_run':
                proposal = dict(proposed_skill='dynamic/run', reason='diagnostic fixed run', neighbors=[])
                chosen = dict(executed_skill='dynamic/run', fallback=False,
                              reason='diagnostic fixed run; initiation eligibility still enforced')
            decision.update(observation=item, pattern=firing, proposal=proposal,
                            supervision=chosen, eligibility=eligibility)
            write_json(folder/'decision.json', decision)
            event('decision', **decision)
            return chosen['executed_skill'].removeprefix('dynamic/') if chosen['executed_skill'] else None

        def sample(model, data, skill):
            nonlocal next_sample, renderer
            if data.time < next_sample:
                return
            next_sample = float(data.time)+.1
            rot = data.body('base').xmat.reshape(3, 3)
            observation = dict(time=float(data.time), position=data.qpos[:2].tolist(),
                speed=float(np.linalg.norm(data.qvel[:2])), height=float(data.qpos[2]),
                yaw=float(np.arctan2(rot[1, 0], rot[0, 0])), upright=float(rot[2, 2]))
            item = observe(case.scenario, observation, case.jump_requested,
                observation_id=f'{case.scenario.name}-{data.time:.3f}', context_id=case.scenario.name)
            firing = encoder.encode(item['features'])
            phase = ('settling' if not decision else
                     'executing' if data.time < decision['observation']['time_s']+duration else 'recovering')
            terrain_contacts = []
            for contact in data.contact:
                names = [model.geom(int(g)).name or str(g) for g in (contact.geom1, contact.geom2)]
                if any(name.startswith('terrain_') for name in names):
                    terrain_contacts.append(names)
            heights = [p.surface_height(*data.qpos[:2]) for p in case.scenario.terrain]
            latest.update(time=float(data.time), phase=phase, active_skill=skill,
                observation=item, diagnostic_pattern=firing, qpos=data.qpos.tolist(),
                qvel=data.qvel.tolist(), actuator_torque=data.ctrl.tolist(),
                terrain_contacts=terrain_contacts,
                support_height_under_base=max([0.]+[h for h in heights if h is not None]))
            telemetry.write(json.dumps(latest, allow_nan=False)+'\n'); telemetry.flush()
            if movie is not None:
                import mujoco
                from PIL import Image, ImageDraw
                if renderer is None:
                    renderer = mujoco.Renderer(model, height=480, width=640)
                cam = mujoco.MjvCamera()
                cam.lookat[:] = [*data.qpos[:2], .2]
                cam.distance, cam.azimuth, cam.elevation = 4., np.rad2deg(observation['yaw'])+100, -30
                if skill=='jump' and jump_profile=='smooth_forward':
                    cam.distance=2.4
                renderer.update_scene(data, camera=cam)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 65), fill='black')
                draw.text((8, 6), f'{case.scenario.name} | {phase} | {skill} | t={data.time:.1f}s\n'
                          f"Current counts: {firing['spike_counts']}", fill='white')
                movie.append_data(np.asarray(frame))

        def overlay(view, model, data, skill):
            nonlocal last_overlay
            import mujoco
            if not latest or latest['time'] == last_overlay:
                return
            last_overlay = latest['time']
            view.cam.lookat[:] = [*data.qpos[:2], .2]
            view.cam.distance, view.cam.azimuth, view.cam.elevation = 4., np.rad2deg(case.scenario.start_yaw)+100, -30
            if skill=='jump' and jump_profile=='smooth_forward':
                view.cam.distance=2.4
            selected = decision.get('supervision', {})
            text = (f'{case.scenario.name} | {latest["phase"]} | t={data.time:.1f}s\n'
                    f'Actual skill: {skill} | selected: {selected.get("executed_skill")}\n'
                    f'Proposal: {decision.get("proposal", {}).get("proposed_skill")}\n'
                    f'Reason: {selected.get("reason", "settling before decision")}\n'
                    f'Decision counts: {decision.get("pattern", {}).get("spike_counts", [])}\n'
                    f'Current counts: {latest["diagnostic_pattern"]["spike_counts"]}\n'
                    f'Position: {data.qpos[0]:.2f}, {data.qpos[1]:.2f} m | height: {data.qpos[2]:.2f} m\n'
                    'Current patterns are diagnostic; one skill per episode. Close to stop.')
            view.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ''))

        try:
            history, summary = execute_dynamic_skill(scenario=case.scenario, seed=case.scene_seed,
                selector=select, viewer=viewer, frame_callback=sample, viewer_callback=overlay,
                timeout=timeout, interrupt_after=interrupt_after, duration=duration,
                jump_profile=jump_profile,jump_config=jump_config)
            measured = measure_case_outcome(case, history, summary)
            result = dict(context_id=case.scenario.name, summary=summary, outcome=measured,
                          decision=decision, association_checksum=memory.checksum,
                          selection_mode=selection,
                          encoder_checksum=encoder.checksum)
            save_trajectory(folder/'trajectory.csv', history)
            write_json(folder/'result.json', result)
            event('finish', summary=summary, outcome=measured)
            return result
        except BaseException as error:
            event('error', error_type=type(error).__name__, message=str(error))
            raise
        finally:
            if renderer is not None:
                renderer.close()
            if movie is not None:
                movie.close()


def run(*, output=None, association=DEFAULT_ASSOCIATION, cases=None,
        evaluate=False, viewer=False, repeat=False, video=False):
    encoder = RandomLIFEncoder.load(ENCODER_PATH)
    memory = AssociationMemory.load(association, encoder.checksum)
    selected = ([case for case in situations() if case.variant >= 5] if evaluate else
                [next(c for c in situations() if c.scenario.name == name)
                 for name in (cases or ['clear-00', 'clear-01'])])
    if repeat and (not viewer or evaluate):
        raise ValueError('Repeat requires a live viewer and demo cases')
    folder = Path(output) if output else ROOT/'output/spike_pattern_skills/executions'/uuid.uuid4().hex
    if not folder.is_absolute():
        folder = ROOT/folder
    folder.mkdir(parents=True, exist_ok=False)
    results = []
    while True:
        for case in selected:
            result = execute_case(case, memory, encoder, folder/f'{len(results):03d}-{case.scenario.name}',
                                  viewer=viewer, video=video)
            results.append(result)
            aggregate = dict(episodes=len(results), task_fulfilled=sum(r['outcome']['task_fulfilled'] for r in results),
                primitive_completed=sum(r['summary']['success'] for r in results),
                contacts=sum(r['summary']['collision'] for r in results), falls=sum(r['summary']['fall'] for r in results),
                fallbacks=sum(r['decision'].get('supervision', {}).get('fallback', True) for r in results),
                executed_skills=dict(Counter(r['summary']['skill'] for r in results)),
                max_torque=max(r['summary']['max_torque'] for r in results),
                max_torque_limit_ratio=max(r['summary']['max_torque_limit_ratio'] for r in results),
                encoder_checksum=encoder.checksum, association_checksum=memory.checksum,
                split='held_out' if evaluate else 'demo')
            write_json(folder/'aggregate.json', aggregate)
            print(case.scenario.name, result['summary']['skill'], result['summary']['reason'],
                  result['outcome'], flush=True)
            if result['summary']['reason'] == 'viewer closed':
                return
        if not repeat:
            print(json.dumps(aggregate, indent=2), flush=True)
            return aggregate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--association', type=Path, default=DEFAULT_ASSOCIATION)
    parser.add_argument('--cases', nargs='+')
    for flag in ('evaluate', 'viewer', 'repeat', 'video'):
        parser.add_argument('--'+flag, action='store_true')
    run(**vars(parser.parse_args()))
