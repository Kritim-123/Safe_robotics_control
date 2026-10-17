"""Dispatch using saved model schema, independently of user-facing arm names."""
import json
import numpy as np
from experiments.continual_snn.loading import load_selector
from .models import Selector
from .factored import FactoredGoalSelector


def load_selector_any(path):
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata'].item()))
    if metadata['kind']=='factored_goal_count_association':return FactoredGoalSelector.load(path)
    return Selector.load(path) if metadata['kind']=='terrain_goal_selector' else load_selector(path)
