"""Skill requests and physically permitted handoffs are different events."""
from dataclasses import asdict, dataclass
import math

SKILLS = ('wait', 'run', 'jump')


@dataclass(frozen=True)
class HandoffConfig:
    decision_dt: float = .1
    confirmations: int = 2
    minimum_dwell: float = .5
    stable_hold: float = .2
    stale_after: float = .35

    def validate(self):
        if (type(self.confirmations) is not int or not 1 <= self.confirmations <= 10
                or not all(math.isfinite(v) for k,v in asdict(self).items() if k!='confirmations')
                or self.decision_dt <= 0 or min(self.minimum_dwell,self.stable_hold)<0
                or self.stale_after <= self.decision_dt):
            raise ValueError('Invalid handoff configuration')


class HandoffSupervisor:
    """Pure logic, no physics writes. One instance per physical episode.

    Requests are debounced using labels from *fresh independent packets*.
    A same-skill observation never restarts it. Geometric request masks are
    distinct from stable-state entry masks. Braking uses the existing gait.
    """
    def __init__(self, config=None):
        self.config=config or HandoffConfig();self.config.validate()
        self.current=self.desired='wait'
        self.mode='settling'
        self.entered_at=None
        self.pending=None;self.pending_count=0
        self.last_observation=None;self.last_advance=None
        self.stable_since=None
        self.jump_consumed=False
        self.switches=0
        self.events=[]

    def _event(self, now, kind, **values):
        item=dict(time=float(now),event=kind,current=self.current,mode=self.mode,**values)
        self.events.append(item)
        return item

    def start(self, now):
        if self.entered_at is not None or not math.isfinite(now) or now<0:
            raise ValueError('Supervisor already started or invalid start time')
        self.entered_at=float(now);self.mode='active'
        self._event(now,'started')

    def observe(self, now, proposed_skill, request_mask, *, jump_requested=False,
                observation_time=None, valid=True):
        if self.mode=='failed':raise RuntimeError('Failed supervisor requires a new episode')
        if not math.isfinite(now) or now<0 or self.entered_at is None:
            raise ValueError('Observation requires a started supervisor and valid time')
        if proposed_skill is not None:
            proposed_skill=proposed_skill.removeprefix('dynamic/')
            if proposed_skill not in SKILLS:
                raise ValueError('Unknown requested skill')
        stamp=now if observation_time is None else observation_time
        bad=(not valid or not math.isfinite(stamp) or stamp>now+1e-9
             or now-stamp>self.config.stale_after)
        if self.last_observation is not None and stamp<=self.last_observation:
            if stamp==self.last_observation and not bad:
                return self._event(now,'duplicate_packet',desired=self.desired)
            bad=True
        if not jump_requested:
            self.jump_consumed=False
        proposed=proposed_skill
        reason='proposal'
        if bad:
            proposed='wait';reason='invalid_or_stale_packet'
        elif proposed is None:
            proposed='wait';reason='unknown_pattern'
        elif not request_mask.get(proposed,False):
            proposed='wait';reason='request_ineligible'
        elif proposed=='jump' and (not jump_requested or (self.jump_consumed and self.current!='jump')):
            proposed='wait';reason='jump_request_absent_or_consumed'
        if not bad:
            self.last_observation=float(stamp)
        if proposed==self.pending:
            self.pending_count+=1
        else:
            self.pending=proposed;self.pending_count=1
        # Invalid/stale input requests braking immediately. Normal classification
        # changes need confirmations; no sensor-sample count is carried over.
        if bad or self.pending_count>=self.config.confirmations:
            self.desired=proposed
        return self._event(now,'decision_packet',proposal=proposed_skill,accepted_request=proposed,
            desired=self.desired,reason=reason,confirmation_count=self.pending_count)

    def _enter(self, now, skill):
        previous=self.current
        self.current=skill;self.mode='active';self.entered_at=float(now);self.stable_since=None
        if skill=='jump':
            self.jump_consumed=True
        self.switches+=int(skill!=previous)
        return self._event(now,'handoff',previous=previous,next=skill)

    def _checked_entry(self,now,wanted,entry_mask):
        chosen=wanted if entry_mask.get(wanted,False) else 'wait' if entry_mask.get('wait',False) else None
        if chosen is None:
            self.mode='failed'
            return self._event(now,'no_eligible_fallback',wanted=wanted)
        return self._enter(now,chosen)

    def advance(self, now, *, stable, entry_mask, continuation_mask=None, jump_finished=False):
        if self.mode=='failed':raise RuntimeError('Failed supervisor requires a new episode')
        if self.entered_at is None or not math.isfinite(now) or now<self.entered_at:
            raise ValueError('Invalid supervisor clock')
        if self.last_advance is not None and now<self.last_advance:
            raise ValueError('Supervisor clock moved backwards')
        self.last_advance=float(now)
        continuation_mask=continuation_mask or entry_mask
        if self.last_observation is None or now-self.last_observation>self.config.stale_after:
            self.desired='wait'
        if self.current=='jump' and self.mode=='active':
            if not jump_finished:
                return None
            # A fresh jump requires the external request to be released first.
            wanted=self.desired if self.desired!='jump' else 'wait'
            return self._checked_entry(now,wanted if stable else 'wait',entry_mask)
        if self.mode=='braking':
            self.stable_since=(now if self.stable_since is None else self.stable_since) if stable else None
            if self.stable_since is not None and now-self.stable_since>=self.config.stable_hold:
                return self._checked_entry(now,self.desired,entry_mask)
            return None
        if self.current=='wait' and not continuation_mask.get('wait',False):
            self.mode='failed'
            return self._event(now,'no_eligible_fallback',wanted=self.desired)
        if self.current=='run' and (self.desired!='run' or not continuation_mask.get('run',False)):
            self.mode='braking';self.stable_since=None
            return self._event(now,'braking',pending_request=self.desired)
        if self.current==self.desired:
            return None
        if now-self.entered_at<self.config.minimum_dwell:
            return None
        if stable and entry_mask.get(self.desired,False):
            return self._enter(now,self.desired)
        return None
