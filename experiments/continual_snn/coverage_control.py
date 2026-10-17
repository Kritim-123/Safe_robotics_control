"""Matched V1-prototype control: same neurons/decoder, older association data."""
import hashlib
import json
from .models import ROOT
from .loading import load_selector
from .prototypes import PrototypeMemory
from .scenes import cases
from .stress_scenes import stress_cases
from .observations import SensorConfig
from .evaluate import verify,DEADLINE
from .study import run_batch,save


def main():
    base=ROOT/'output/continual_snn';freeze=json.loads((base/'coverage-v2-freeze.json').read_text())
    old_path=ROOT/'docs/agents/continual_snn_v1/association_memory.npz'
    new_path=ROOT/freeze['checkpoint']
    old,new=load_selector(old_path),load_selector(new_path)
    if not isinstance(old,PrototypeMemory) or not isinstance(new,PrototypeMemory):raise ValueError('Prototype control requires matched model types')
    if (old.encoder.checksum!=new.encoder.checksum or old.neighbors!=new.neighbors
            or old.max_distance!=new.max_distance):raise ValueError('Control changed more than association data')
    scenes=dict(stress=stress_cases(range(11200,11210)),regular=cases('confirmation',range(11300,11310)),
                noise=cases('confirmation',range(11300,11304)))
    for name,items in scenes.items():
        if json.dumps([c.to_dict() for c in items],sort_keys=True)!=json.dumps(freeze['cases'][name],sort_keys=True):
            raise ValueError('V2 matched case manifests differ')
    save(base/'coverage-v2-control-protocol.json',dict(
        status='Additional matched model-family control; same frozen V2 cases, not additional independent scenes',
        reason='Isolate association-data expansion from readout-versus-prototype and neuron-count changes',
        encoder_checksum=old.encoder.checksum,neighbors=old.neighbors,max_distance=old.max_distance,
        old_model_checksum=old.checksum,new_model_checksum=new.checksum,
        old_training_rows=old.training_rows,new_training_rows=new.training_rows,
        old_patterns=len(old.counts),new_patterns=len(new.counts),
        old_checkpoint_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((ROOT/'experiments/continual_snn/coverage_control.py').read_bytes()).hexdigest(),
        source_recorded_before_control_outcomes=True))
    for phase,items in scenes.items():
        verify();sensor=SensorConfig(.03,.015,.1,20261019) if phase=='noise' else None
        run_batch(base/('coverage-v2-control-'+phase),items,[('v1_prototype80','continuous',old)],
                  sensor=sensor,deadline=DEADLINE)


if __name__=='__main__':main()
