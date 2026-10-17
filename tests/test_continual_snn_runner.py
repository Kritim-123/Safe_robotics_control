from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch

import numpy as np

from experiments.continual_snn.models import ROOT
from experiments.continual_snn.runner import execute
from experiments.continual_snn.scenes import make_case
from experiments.continual_snn.observations import SensorConfig


class ContinuousPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=ROOT/'output/continual_snn/test-tmp';cls.parent.mkdir(parents=True,exist_ok=True)

    def test_multiple_checked_handoffs_without_state_resets(self):
        case=make_case('run_jump_run',4004)
        seen=[];data_ids=[]
        import mujoco
        real_reset=mujoco.mj_resetDataKeyframe
        resets=[]
        def reset(model,data,key):
            resets.append(id(data));return real_reset(model,data,key)
        def frame(model,data,supervisor):
            seen.append(np.r_[data.time,data.qpos[:3]])
            data_ids.append(id(data))
            np.testing.assert_array_equal(data.xfrc_applied,0.)
            np.testing.assert_array_equal(data.qfrc_applied,0.)
            np.testing.assert_allclose(data.actuator_force,data.ctrl,atol=1e-10)
        with tempfile.TemporaryDirectory(dir=self.parent) as folder:
            with patch('experiments.continual_snn.runner.mujoco.mj_resetDataKeyframe',side_effect=reset):
                result=execute(case,Path(folder)/'episode',mode='rule',frame_callback=frame,full_trace=True)
            self.assertTrue(result['task_fulfilled'],result)
            self.assertEqual(result['completed_jumps'],1)
            self.assertGreaterEqual(result['switches'],5)
            self.assertLessEqual(result['max_torque'],28.)
            self.assertLessEqual(result['max_torque_limit_ratio'],1.)
            self.assertFalse(result['collision'] or result['fall'])
            events=[json.loads(line) for line in (Path(folder)/'episode/events.jsonl').read_text().splitlines()]
            for event in events:
                if event['event']=='handoff':
                    self.assertLess(np.linalg.norm(event['qvel'][:3]),.04)
                    self.assertLess(np.linalg.norm(event['qvel'][3:6]),.2)
            self.assertEqual(resets.count(data_ids[0]),1)
        points=np.asarray(seen)
        self.assertTrue(np.all(np.diff(points[:,0])>0))
        self.assertLess(np.max(np.linalg.norm(np.diff(points[:,1:],axis=0),axis=1)),.03)

    def test_sustained_request_causes_one_hop_and_stale_input_waits(self):
        with tempfile.TemporaryDirectory(dir=self.parent) as folder:
            result=execute(make_case('sustained_jump',4004),Path(folder)/'hop',mode='rule')
            self.assertEqual(result['completed_jumps'],1)
            self.assertTrue(result['task_fulfilled'])
            stale=execute(make_case('clear_stop',4004),Path(folder)/'stale',mode='rule',
                          sensor_config=SensorConfig(delay=.5))
            self.assertEqual(stale['completed_jumps'],0)
            self.assertFalse(stale['task_fulfilled'])
            self.assertLess(abs(stale['progress']),.02)
            self.assertFalse(stale['collision'] or stale['fall'])

    def test_observer_exception_releases_torques_and_early_viewer_close(self):
        held=[]
        def fail(model,data,supervisor):
            if data.time>3.:
                held.append(data)
                raise RuntimeError('observer failed')
        with tempfile.TemporaryDirectory(dir=self.parent) as folder:
            with self.assertRaisesRegex(RuntimeError,'observer failed'):
                execute(make_case('clear_stop',4004),Path(folder)/'failed',mode='rule',frame_callback=fail)
            np.testing.assert_array_equal(held[0].ctrl,0.)
            view=MagicMock();view.is_running.return_value=False
            context=MagicMock();context.__enter__.return_value=view
            with patch('experiments.continual_snn.runner.mujoco.viewer.launch_passive',return_value=context):
                result=execute(make_case('clear_stop',4004),Path(folder)/'closed',mode='rule',viewer=True)
            self.assertEqual(result['reason'],'viewer closed')
            self.assertFalse(result['task_fulfilled'])
            self.assertTrue(np.isfinite(result['minimum_clearance']))

    def test_fresh_request_can_cancel_preparation_before_thrust(self):
        from experiments.continual_snn.stress_scenes import make_stress_case
        case=make_stress_case('cancel_preparation',8200)
        with tempfile.TemporaryDirectory(dir=self.parent) as folder:
            result=execute(case,Path(folder)/'cancel',mode='rule')
            self.assertTrue(result['task_fulfilled'],result)
            self.assertEqual(result['completed_jumps'],0)
            self.assertEqual(result['failed_jumps'],0)
            self.assertEqual(result['cancelled_jumps'],1)
            self.assertFalse(result['collision'] or result['fall'])
            events=[json.loads(s) for s in (Path(folder)/'cancel/events.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['event']=='jump_preparation_cancelled' for e in events))

    def test_strict_original_schema_and_predictive_schema_stay_separate(self):
        from experiments.continual_snn.models import FrozenBankModel
        from experiments.continual_snn.predictive import PredictivePacketEncoder
        from experiments.continual_snn.readout import SpikeReadout
        from experiments.continual_snn.data import static_records
        rows=static_records()[:12]
        features=[dict(r['features'],predicted_run_clearance=1.,predicted_jump_clearance=1.) for r in rows]
        learned=SpikeReadout(PredictivePacketEncoder()).fit(features,[r['label'] for r in rows],[r['group'] for r in rows])
        case=replace(make_case('clear_stop',4004),finish_after=5.,timeout=8.)
        with tempfile.TemporaryDirectory(dir=self.parent) as folder:
            for name,model,dimensions,neurons in (('old',FrozenBankModel(),22,16),('predictive',learned,24,80)):
                output=Path(folder)/name
                result=execute(case,output,model=model)
                self.assertFalse(result['collision'] or result['fall'])
                first=json.loads((output/'decisions.jsonl').read_text().splitlines()[0])
                self.assertEqual(len(first['features']),dimensions)
                self.assertEqual(len(first['proposal']['spike_counts']),neurons)


if __name__=='__main__':unittest.main()
