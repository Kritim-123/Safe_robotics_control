"""Train spiking-output LIF on outcome-labelled, episode-grouped MuJoCo data."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from hybrid.skill_lif import FEATURES, SKILLS, SkillLIF, decode_counts


def load_contexts(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    grouped = {}
    for row in rows:
        grouped.setdefault(row['context_id'], []).append(row)
    contexts = []
    for key, group in grouped.items():
        if len({r['split'] for r in group}) != 1 or len({r['label_action'] for r in group}) != 1:
            raise ValueError(f'Inconsistent group {key}')
        if len(group) != len(SKILLS) or {r['chosen_action'] for r in group} != set(SKILLS):
            raise ValueError(f'Incomplete candidate trials {key}')
        reference = group[0]
        if reference['split'] not in ('train','validation','test'):
            raise ValueError('Unknown split')
        if any(r[f'allowed_{a}'] not in ('0','1') for r in group for a in SKILLS):
            raise ValueError('Eligibility must be binary')
        if any(tuple(r[f'allowed_{a}'] for a in SKILLS) != tuple(reference[f'allowed_{a}'] for a in SKILLS) for r in group):
            raise ValueError('Sibling eligibility mismatch')
        if not reference['label_action']:
            continue
        label = SKILLS.index(reference['label_action'])
        winner = next(r for r in group if r['chosen_action'] == SKILLS[label])
        if winner['outcome_success'] != '1' or winner[f'allowed_{SKILLS[label]}'] != '1':
            raise ValueError('Failed or ineligible action cannot be a training target')
        x = np.array([float(reference[f]) for f in FEATURES], np.float32)
        mask = np.array([reference[f'allowed_{s}'] == '1' for s in SKILLS])
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite training features')
        for row in group[1:]:
            if not np.allclose(x, [float(row[f]) for f in FEATURES], rtol=0, atol=1e-6):
                raise ValueError('Counterfactual siblings have different initial observations')
        contexts.append((key, reference['split'], x, mask, label))
    return contexts


def train(dataset, output, epochs=200, seed=37):
    import torch
    from torch import nn
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    contexts = load_contexts(dataset)
    partitions = {s: [c for c in contexts if c[1] == s] for s in ('train','validation','test')}
    if any(not v for v in partitions.values()):
        raise ValueError('All three group-disjoint splits need labelled contexts')
    raw = np.stack([c[2] for c in partitions['train']])
    mean, std = raw.mean(0), np.maximum(raw.std(0), 1e-3)
    def tensors(split):
        data = partitions[split]
        return (torch.tensor(np.clip((np.stack([c[2] for c in data])-mean)/std,-5,5)),
                torch.tensor([c[4] for c in data]), np.stack([c[3] for c in data]))
    x, y, masks = tensors('train')
    vx, vy, vm = tensors('validation')
    if len(np.unique(y.numpy())) != len(SKILLS):
        raise ValueError('Actual successful training outcomes must cover every declared output skill')

    class Spike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(value)
            return (value >= 0).float()
        @staticmethod
        def backward(ctx, grad):
            (value,) = ctx.saved_tensors
            return grad/(1+5*value.abs()).square()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.fc2 = nn.Linear(len(FEATURES),32), nn.Linear(32,len(SKILLS))
            nn.init.constant_(self.fc1.bias,.25)
            nn.init.constant_(self.fc2.bias,.25)
        def forward(self, x):
            current = self.fc1(x)
            v1 = torch.zeros_like(current)
            v2 = torch.zeros((len(x),len(SKILLS)),dtype=x.dtype)
            counts = torch.zeros_like(v2)
            for _ in range(16):
                v1 = .9*v1+current
                s1 = Spike.apply(v1-1)
                v1 = v1-s1.detach()
                v2 = .9*v2+self.fc2(s1)
                s2 = Spike.apply(v2-1)
                v2 = v2-s2.detach()
                counts = counts+s2
            return counts

    net = Net()
    optimizer = torch.optim.Adam(net.parameters(),lr=.004)
    frequencies = np.bincount(y.numpy(),minlength=len(SKILLS))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(y)/(len(SKILLS)*frequencies),dtype=torch.float32))
    best, best_validation_loss, best_state, curve = -1., float('inf'), None, []
    for epoch in range(epochs):
        optimizer.zero_grad()
        counts = net(x)
        loss = loss_fn(counts,y)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),5)
        optimizer.step()
        with torch.no_grad():
            validation_counts = net(vx)
            validation_loss = float(nn.functional.cross_entropy(validation_counts,vy))
            vc = validation_counts.numpy()
        choices = [decode_counts(c,m)[0] for c,m in zip(vc,vm)]
        accuracy = float(np.mean(np.array(choices,dtype=object)==vy.numpy()))
        curve.append(dict(epoch=epoch+1,loss=float(loss.detach()),validation_accuracy=accuracy,validation_loss=validation_loss))
        if accuracy > best or (accuracy == best and validation_loss < best_validation_loss):
            best, best_validation_loss, best_state = accuracy, validation_loss, copy.deepcopy(net.state_dict())
        if (epoch+1)%25 == 0:
            print(json.dumps(curve[-1]),flush=True)
    net.load_state_dict(best_state)
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    payload = dict(mean=mean,std=std,steps=np.array(16),beta=np.array(.9),
                   feature_names=np.array(FEATURES),skill_names=np.array(SKILLS))
    for i,layer in enumerate((net.fc1,net.fc2),1):
        payload[f'w{i}'],payload[f'b{i}'] = layer.weight.detach().numpy(),layer.bias.detach().numpy()
    path = output/'model.npz'
    np.savez_compressed(path,**payload)
    deployed = SkillLIF(path)
    metrics = {}
    for split, group in partitions.items():
        tx,ty,tm = tensors(split)
        with torch.no_grad():
            torch_counts = net(tx).numpy()
        numpy_counts = np.stack([deployed.predict(c[2])[0] for c in group])
        if not np.array_equal(torch_counts,numpy_counts):
            raise RuntimeError('Exported spike counts differ from training implementation')
        choices = [decode_counts(c,m)[0] for c,m in zip(numpy_counts,tm)]
        matrix = np.zeros((len(SKILLS),len(SKILLS)+1),int)
        for label,pred in zip(ty.numpy(),choices):
            matrix[label,len(SKILLS) if pred is None else pred] += 1
        metrics[split] = dict(contexts=len(group),accuracy=float(np.mean(np.array(choices,dtype=object)==ty.numpy())),
                              label_counts=np.bincount(ty.numpy(),minlength=len(SKILLS)).tolist(),
                              confusion=matrix.tolist(),export_exact=True,
                              unmasked_accuracy=float(np.mean(numpy_counts.argmax(1)==ty.numpy())),
                              zero_eligible_spikes=sum(not np.any(c[m]>0) for c,m in zip(numpy_counts,tm)),
                              tied_eligible_maxima=int(sum(np.sum(c[m]==np.max(c[m]))>1 for c,m in zip(numpy_counts,tm) if m.any())))
    report = dict(seed=seed,epochs=epochs,best_validation_accuracy=best,metrics=metrics,curve=curve,
                  dataset_sha256=hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  feature_names=FEATURES,skill_names=SKILLS,output='binary LIF spikes; eligible spike-count decoding',
                  checkpoint_selection='highest validation accuracy then lowest validation cross entropy; test never selects checkpoint',
                  training='surrogate-gradient cross entropy on output spike counts; train-only normalization',
                  torch_version=torch.__version__,numpy_version=np.__version__)
    (output/'training.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(metrics,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('output/skill_lif/dataset/decisions.csv'))
    parser.add_argument('--output',type=Path,default=Path('output/skill_lif/model'))
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--seed',type=int,default=37)
    args=parser.parse_args()
    if args.epochs < 1: parser.error('epochs must be positive')
    train(args.dataset,args.output,args.epochs,args.seed)


if __name__=='__main__':
    main()
