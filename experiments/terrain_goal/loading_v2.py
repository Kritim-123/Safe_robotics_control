"""Superset loader; the frozen V1 reader is retained unchanged."""
import json
import numpy as np
from .loading import load_selector_any
from .hysteresis import HystereticGoalSelector


def load_any(path):
    with np.load(path,allow_pickle=False) as b:meta=json.loads(str(b['metadata'].item()))
    return HystereticGoalSelector.load(path) if meta['kind']=='goal_count_hysteresis_v2' else load_selector_any(path)
