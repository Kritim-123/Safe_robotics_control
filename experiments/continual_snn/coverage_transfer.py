"""Post-hoc V2 transfer arm on unchanged V1 scene/sensor manifests."""
import hashlib
import json
from .models import ROOT
from .loading import load_selector
from .evaluate import plan,verify,DEADLINE
from .study import run_batch,save


def main():
    base=ROOT/'output/continual_snn';path=ROOT/'docs/agents/continual_snn_v2/primary.npz'
    model=load_selector(path);checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    phases=('terrain','heading','severe_noise','stale')
    save(base/'coverage-v2-transfer-protocol.json',dict(
        status='Post-hoc matched transfer checks on the original V1 manifests, not independent extra environments',
        model_checksum=model.checksum,checkpoint_sha256=checksum,phases=list(phases),
        no_training_or_parameter_changes=True,original_failed_cases_and_acceptance_criteria_retained=True,
        source_sha256=hashlib.sha256((ROOT/'experiments/continual_snn/coverage_transfer.py').read_bytes()).hexdigest()))
    for phase in phases:
        verify()
        if hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:raise ValueError('V2 checkpoint changed')
        scenes,_,sensor=plan()[phase]
        run_batch(base/('coverage-v2-transfer-'+phase),scenes,[('v2_selected','continuous',model)],
                  sensor=sensor,deadline=DEADLINE)


if __name__=='__main__':main()
