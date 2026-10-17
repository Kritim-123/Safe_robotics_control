"""Load the frozen primary candidates or the losslessly compacted legacy bank."""
import json
import numpy as np
from .models import load_model


def load_selector(path):
    with np.load(path,allow_pickle=False) as bundle:
        kind=json.loads(str(bundle['metadata'].item()))['kind']
    if kind=='compact_legacy_count_bank':
        from .compact_bank import CompactLegacyBank
        return CompactLegacyBank.load(path)
    return load_model(path)
