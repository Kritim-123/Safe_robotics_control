from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from hybrid.arena import Arena, Scenario, build_model
from hybrid.terrain import TerrainPatch
from experiments.spike_pattern_skills.associations import AssociationMemory
from experiments.spike_pattern_skills.curriculum import STAGES, curriculum
from experiments.spike_pattern_skills.encoder import RandomLIFEncoder
from experiments.spike_pattern_skills.execute_associations import DEFAULT_ASSOCIATION, execute_case
from experiments.spike_pattern_skills.learn_associations import ENCODER_PATH
from experiments.spike_pattern_skills.run_curriculum import assessment, run as run_curriculum


class TerrainCurriculumTests(unittest.TestCase):
    def test_protocol_and_reproducibility(self):
        cases = curriculum()
        self.assertEqual(cases, curriculum())
        self.assertEqual(len(cases), 60)
        self.assertEqual(len({c.scenario.name for c in cases}), 60)
        for stage in range(10):
            group = [c for c in cases if c.stage == stage]
            self.assertEqual([c.split for c in group], ['development']*3+['validation']+['held_out']*2)
            self.assertTrue(all(not c.jump_requested for c in group))
        self.assertNotEqual(curriculum()[7].scenario, curriculum(20261018)[7].scenario)
        with self.assertRaises(ValueError):
            curriculum(-1)

    def test_family_difficulty_parameters_increase(self):
        cases = curriculum()
        for stage in (1, 2, 3, 5):
            heights = [c.scenario.terrain[0].height for c in cases if c.stage == stage]
            self.assertTrue(all(a < b for a, b in zip(heights, heights[1:])))
        frictions = [c.scenario.terrain[0].friction for c in cases if c.stage == 6]
        self.assertTrue(all(a > b for a, b in zip(frictions, frictions[1:])))
        speeds = [c.scenario.obstacles[-1].velocity[1] for c in cases if c.stage == 8]
        self.assertTrue(all(a < b for a, b in zip(speeds, speeds[1:])))

    def test_all_sixty_models_compile_and_support_is_not_obstacle(self):
        for case in curriculum():
            model = build_model(case.scenario)
            data = mujoco.MjData(model)
            arena = Arena(case.scenario, model)
            self.assertEqual(len(arena.terrain_geoms), len(case.scenario.terrain))
            self.assertEqual(len(arena.sensed_obstacles(data)), len(case.scenario.obstacles))
            self.assertFalse(arena.terrain_geoms & arena.hazard_geoms)

    def test_invalid_terrain_rejected(self):
        valid = TerrainPatch('test', (1, 0), (.2, .5), .02)
        for patch in (replace(valid, height=0), replace(valid, height=float('nan')),
                      replace(valid, half_extents=(-1, 1)), replace(valid, kind='gap'),
                      replace(valid, friction=0), replace(valid, base_height=-1),
                      replace(valid, name='bad name'), replace(valid, center=(10, 0))):
            with self.assertRaises(ValueError):
                Scenario('bad', terrain=(patch,)).validate()
        with self.assertRaises(ValueError):
            Scenario('duplicate', terrain=(valid, valid)).validate()

    def test_json_roundtrip_and_legacy_serialization(self):
        self.assertNotIn('terrain', Scenario('flat').to_dict())
        spec = curriculum()[-1].scenario
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'scene.json'
            path.write_text(json.dumps(spec.to_dict()))
            loaded = Scenario.load(path)
            self.assertEqual(json.dumps(loaded.to_dict(), sort_keys=True), json.dumps(spec.to_dict(), sort_keys=True))

    def test_ramp_height_is_actual_collidable_sloping_surface(self):
        spec = curriculum()[12].scenario
        model = build_model(spec)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        patch = spec.terrain[0]
        heights = []
        for delta in (-.15, .15):
            x, y = patch.center[0]+delta, .6
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(model, data, np.array([x, y, 1.]), np.array([0., 0., -1.]),
                                     None, 1, -1, hit)
            self.assertEqual(model.geom(int(hit[0])).name, 'terrain_'+patch.name)
            self.assertAlmostEqual(1-distance, patch.surface_height(x, y), places=6)
            heights.append(1-distance)
        self.assertGreater(heights[1], heights[0])

    def test_contact_friction_and_foot_support(self):
        spec = Scenario('contact', terrain=(TerrainPatch('patch', (0, 0), (1, 1), .01, friction=.3),))
        model = build_model(spec)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
        data.qpos[2] += .022
        mujoco.mj_forward(model, data)
        arena = Arena(spec, model)
        contacts = [c for c in data.contact if c.geom1 in arena.terrain_geoms or c.geom2 in arena.terrain_geoms]
        self.assertEqual(len(contacts), 4)
        for contact in contacts:
            np.testing.assert_allclose(contact.friction[:2], [.3, .3])
            self.assertEqual(contact.dim, 6)
            np.testing.assert_allclose(contact.solimp, model.geom('FL').solimp)
            np.testing.assert_allclose(contact.solref, model.geom('FL').solref)
        self.assertFalse(arena.collided(data))
        self.assertFalse(any(model.geom(int(g)).name == 'arena_floor' for c in data.contact for g in (c.geom1, c.geom2)))
        terrain_id = next(iter(arena.terrain_geoms))
        body_id = next(iter(arena.nonfoot_robot_geoms))
        fake = SimpleNamespace(contact=[SimpleNamespace(geom1=terrain_id, geom2=body_id)])
        self.assertTrue(arena.collided(fake))

    def test_rotated_surface_height(self):
        patch = TerrainPatch('slope', (1, 0), (.5, 1), .1, kind='ramp', yaw=np.pi/2)
        self.assertAlmostEqual(patch.surface_height(1, -.4), .01)
        self.assertAlmostEqual(patch.surface_height(1, .4), .09)
        self.assertIsNone(patch.surface_height(1, .6))

    def test_gates_require_completion_and_previous_levels(self):
        def rows(stage, passed=True):
            return [dict(mode='association', stage=stage, split=('development' if i < 3 else 'held_out'),
                traversal_success=passed, primitive_completed=True, collision=False, fall=False,
                skill='run' if passed else 'wait', reason='completed') for i in range(6)]
        report = assessment(rows(0)+rows(1, False)+rows(2), ['association'], [0, 1, 2])
        self.assertEqual(report['passed_stages']['association'], [0])
        self.assertEqual(report['groups'][2]['progression_status'], 'locked')
        self.assertEqual(assessment(rows(2), ['association'], [2])['groups'][0]['progression_status'], 'locked')
        self.assertFalse(assessment(rows(0)[:2], ['association'], [0])['groups'][0]['stage_passed'])

    def test_progressive_runner_stops_at_failed_level(self):
        def execute(case, *args, **kwargs):
            return dict(summary=dict(skill='run', reason='completed', success=True, collision=False,
                fall=False, max_torque=20., max_torque_limit_ratio=1.),
                outcome=dict(task_fulfilled=case.stage == 0, final_position=[1.7, 0], body_forward_progress=1.7),
                decision=dict(supervision=dict(fallback=False)))
        with tempfile.TemporaryDirectory() as root, patch(
                'experiments.spike_pattern_skills.run_curriculum.execute_case', side_effect=execute) as controller:
            report = run_curriculum(output=Path(root)/'run', stages=[0, 1, 2],
                                    modes=['direct_run'], progressive=True)
            self.assertEqual(controller.call_count, 12)
            self.assertEqual(report['passed_stages']['direct_run'], [0])
            self.assertEqual(report['groups'][2]['episodes'], 0)
            self.assertEqual(report['groups'][2]['progression_status'], 'locked')
        with self.assertRaises(ValueError):
            run_curriculum(stages=[1], progressive=True)

    def test_actual_flat_traversal_and_terrain_recording(self):
        encoder = RandomLIFEncoder.load(ENCODER_PATH)
        memory = AssociationMemory.load(DEFAULT_ASSOCIATION, encoder.checksum)
        with tempfile.TemporaryDirectory() as root:
            for stage in (0, 1):
                folder = Path(root)/str(stage)
                result = execute_case(curriculum()[stage*6], memory, encoder, folder,
                                      duration=8., timeout=18., selection='direct_run')
                self.assertEqual(result['selection_mode'], 'direct_run')
                if stage == 0:
                    self.assertTrue(result['outcome']['task_fulfilled'])
                    self.assertGreaterEqual(result['outcome']['final_position'][0], 1.55)
                rows = [json.loads(line) for line in (folder/'telemetry.jsonl').read_text().splitlines()]
                self.assertTrue(all('terrain_contacts' in row for row in rows))
                if stage == 1:
                    self.assertTrue(any(row['terrain_contacts'] for row in rows))
                self.assertLessEqual(result['summary']['max_torque_limit_ratio'], 1.)


if __name__ == '__main__':
    unittest.main()
