import unittest

from experiments.continual_snn.supervisor import HandoffSupervisor,HandoffConfig

ALL=dict(wait=True,run=True,jump=True)


class HandoffSupervisorTests(unittest.TestCase):
    def make(self):
        s=HandoffSupervisor();s.start(2.)
        return s

    def propose(self,s,now,skill,**kwargs):
        s.observe(now,skill,ALL,**kwargs)
        s.observe(now+.1,skill,ALL,**kwargs)

    def test_fresh_packets_debounce_and_same_action_does_not_restart(self):
        s=self.make()
        s.observe(2.,'run',ALL)
        self.assertIsNone(s.advance(2.,stable=True,entry_mask=ALL))
        s.observe(2.1,'run',ALL)
        event=s.advance(2.5,stable=True,entry_mask=ALL)
        # The .35s freshness bound has expired by 2.5.
        self.assertIsNone(event)
        self.propose(s,2.5,'run')
        event=s.advance(2.6,stable=True,entry_mask=ALL)
        self.assertEqual(event['next'],'run')
        entered=s.entered_at
        self.propose(s,2.7,'run')
        self.assertIsNone(s.advance(2.8,stable=False,entry_mask=ALL))
        self.assertEqual(s.entered_at,entered)
        self.assertEqual(s.switches,1)

    def test_run_to_jump_brakes_then_revalidates_latest_request(self):
        s=self.make();self.propose(s,2.5,'run');s.advance(2.6,stable=True,entry_mask=ALL)
        self.propose(s,3.,'jump',jump_requested=True)
        self.assertEqual(s.advance(3.1,stable=False,entry_mask=ALL)['event'],'braking')
        self.assertEqual(s.current,'run')
        self.propose(s,3.2,'wait')
        s.advance(3.3,stable=True,entry_mask=ALL)
        self.propose(s,3.4,'wait')
        event=s.advance(3.6,stable=True,entry_mask=ALL)
        self.assertEqual(event['next'],'wait')

    def test_airborne_commitment_and_no_repeated_hop(self):
        s=self.make();self.propose(s,2.5,'jump',jump_requested=True)
        s.advance(2.6,stable=True,entry_mask=ALL)
        self.propose(s,3.,'run')
        self.assertIsNone(s.advance(3.1,stable=False,entry_mask=ALL,jump_finished=False))
        self.assertEqual(s.current,'jump')
        self.propose(s,4.,'jump',jump_requested=True)
        event=s.advance(4.1,stable=True,entry_mask=ALL,jump_finished=True)
        self.assertEqual(event['next'],'wait')
        # The preceding run observations released the external request; emulate
        # a sustained high request after this new jump entry.
        self.assertFalse(s.jump_consumed)

    def test_sustained_jump_request_is_consumed_until_release(self):
        s=self.make();self.propose(s,2.5,'jump',jump_requested=True)
        s.advance(2.6,stable=True,entry_mask=ALL)
        self.assertTrue(s.jump_consumed)
        self.propose(s,3.,'jump',jump_requested=True)
        s.advance(3.1,stable=True,entry_mask=ALL,jump_finished=True)
        self.propose(s,3.2,'jump',jump_requested=True)
        s.advance(3.3,stable=True,entry_mask=ALL)
        self.assertEqual(s.current,'wait')
        s.observe(3.4,'wait',ALL,jump_requested=False)
        self.assertFalse(s.jump_consumed)

    def test_unknown_stale_duplicate_and_episode_isolation(self):
        s=self.make();self.propose(s,2.5,'run');s.advance(2.6,stable=True,entry_mask=ALL)
        before=s.pending_count
        s.observe(2.6,'run',ALL)
        self.assertEqual(s.pending_count,before)
        s.observe(3.,'run',ALL,observation_time=2.)
        self.assertEqual(s.desired,'wait')
        self.assertEqual(s.advance(3.,stable=False,entry_mask=ALL)['event'],'braking')
        t=self.make()
        self.assertIsNone(t.last_observation)
        self.assertEqual(t.current,'wait')
        with self.assertRaises(ValueError):t.observe(2.,'not_a_skill',ALL)
        with self.assertRaises(ValueError):HandoffConfig(confirmations=0).validate()

    def test_braking_does_not_enter_an_ineligible_fallback(self):
        s=self.make();self.propose(s,2.5,'run');s.advance(2.6,stable=True,entry_mask=ALL)
        self.propose(s,3.,'wait');s.advance(3.1,stable=False,entry_mask=ALL)
        s.advance(3.2,stable=True,entry_mask=ALL)
        self.propose(s,3.3,'wait')
        event=s.advance(3.5,stable=True,entry_mask={k:False for k in ALL})
        self.assertEqual(event['event'],'no_eligible_fallback')
        self.assertEqual(s.mode,'failed')

    def test_active_wait_is_not_an_unchecked_fallback_and_failure_is_terminal(self):
        s=self.make();self.propose(s,2.,None)
        event=s.advance(2.1,stable=True,entry_mask={k:False for k in ALL})
        self.assertEqual(event['event'],'no_eligible_fallback')
        with self.assertRaises(RuntimeError):s.observe(2.2,'run',ALL)
        with self.assertRaises(RuntimeError):s.advance(2.2,stable=True,entry_mask=ALL)


if __name__=='__main__':unittest.main()
