"""New walk-up outcome associations; V4 encoder and V3 weights are unchanged."""
from experiments.obstacle_v4.model import JumpHead as CountHead, ROOT, V3, observe

MODEL = ROOT/'docs/agents/obstacle_v5/walk_up_v2.npz'


class WalkUpHead(CountHead):
    @classmethod
    def load(cls, path=MODEL):
        return super().load(path)
