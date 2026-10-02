"""Train the LIF skill selector on independent synthetic geometric encounters.

Supervised imitation of an explicit geometric teacher, not reinforcement
learning and not proof of collision-free closed-loop performance.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from hybrid.perception import FEATURE_NAMES, choose_masked, observe_encounter, teacher_scores
from hybrid.snn import DEFAULT_MODEL, SpikingSelector
from skills.avoidance import SKILL_NAMES


def make_dataset(count, seed):
    rng = np.random.default_rng(seed)
    features, masks, labels = [], [], []
    attempts = 0
    while len(features) < count:
        attempts += 1
        if attempts > count*20:
            raise RuntimeError('Unable to sample enough feasible encounters')
        center = np.array([rng.uniform(0.55, 1.75), rng.uniform(-0.4, 0.4)])
        velocity = np.zeros(2)
        mode = rng.integers(0, 4)
        if mode in (1, 2):
            velocity[1] = rng.choice([-1, 1]) * rng.uniform(0.04, 0.16)
        elif mode == 3:
            velocity[0] = -rng.uniform(0.045, 0.13)
        obstacles = [(center, rng.uniform(0.16, 0.32), velocity)]
        for sign in (-1, 1):
            if rng.random() < 0.3:
                obstacles.append((np.array([rng.uniform(0.4, 1.6), sign*rng.uniform(0.8, 1.4)]),
                                  rng.uniform(0.18, 0.38), np.zeros(2)))
        bounds = (-rng.uniform(0.9, 2), 5, -rng.uniform(1.1, 2.4), rng.uniform(1.1, 2.4))
        observation = observe_encounter(np.zeros(2), rng.uniform(-0.3, 0.3),
                                        np.array([rng.uniform(2.4, 4.0), 0]), obstacles, bounds, 0)
        label = choose_masked(teacher_scores(observation.features), observation.allowed)
        if label is None:
            continue
        features.append(observation.features)
        masks.append(observation.allowed)
        labels.append(label)
    return np.asarray(features, dtype=np.float32), np.asarray(masks), np.asarray(labels, dtype=np.int64)


def train(output=DEFAULT_MODEL.parent, samples=10000, epochs=50, seed=17):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)

    class Spike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(value)
            return (value >= 0).float()

        @staticmethod
        def backward(ctx, grad):
            (value,) = ctx.saved_tensors
            return grad / (1 + 5*value.abs()).square()

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.fc2, self.fc3 = nn.Linear(12, 64), nn.Linear(64, 32), nn.Linear(32, 4)
            nn.init.constant_(self.fc1.bias, 0.25)
            nn.init.constant_(self.fc2.bias, 0.25)

        def forward(self, x):
            current = self.fc1(x)
            m1 = torch.zeros_like(current)
            m2 = torch.zeros((len(x), 32), dtype=x.dtype)
            count = torch.zeros_like(m2)
            for _ in range(16):
                m1 = 0.9*m1 + current
                s1 = Spike.apply(m1 - 1)
                m1 = m1 - s1.detach()
                m2 = 0.9*m2 + self.fc2(s1)
                s2 = Spike.apply(m2 - 1)
                m2 = m2 - s2.detach()
                count = count + s2
            return self.fc3(count / 16)

    started = time.perf_counter()
    raw, masks, labels = make_dataset(samples, seed)
    val_raw, val_masks, val_labels = make_dataset(2000, seed+10000)
    test_raw, test_masks, test_labels = make_dataset(2000, seed+20000)
    mean, std = raw.mean(0), np.maximum(raw.std(0), 1e-3)
    x = torch.from_numpy(np.clip((raw-mean)/std, -5, 5))
    y = torch.from_numpy(labels)
    validation = torch.from_numpy(np.clip((val_raw-mean)/std, -5, 5))
    net = Network()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.004)
    class_counts = np.bincount(labels, minlength=4)
    weights = torch.tensor(np.sqrt(len(labels)/(4*np.maximum(class_counts, 1))), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    best_accuracy, best_state, curve = -1.0, None, []
    for epoch in range(epochs):
        net.train()
        indices = torch.randperm(len(x))
        loss_sum = 0
        for batch in indices.split(256):
            optimizer.zero_grad()
            loss = loss_fn(net(x[batch]), y[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        net.eval()
        with torch.no_grad():
            scores = net(validation).numpy()
        predicted = np.argmax(np.where(val_masks, scores, -np.inf), axis=1)
        accuracy = float(np.mean(predicted == val_labels))
        curve.append(dict(epoch=epoch+1, train_loss=loss_sum/len(x), validation_accuracy=accuracy))
        if accuracy > best_accuracy:
            best_accuracy, best_state = accuracy, copy.deepcopy(net.state_dict())
        if epoch % 5 == 0 or epoch == epochs-1:
            print(json.dumps(curve[-1]), flush=True)
    net.load_state_dict(best_state)
    net.eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(mean=mean, std=std, steps=np.array(16), beta=np.array(0.9),
                   feature_names=np.array(FEATURE_NAMES), skill_names=np.array(SKILL_NAMES))
    for i, layer in enumerate((net.fc1, net.fc2, net.fc3), 1):
        payload[f'w{i}'] = layer.weight.detach().numpy()
        payload[f'b{i}'] = layer.bias.detach().numpy()
    path = output / 'model.npz'
    np.savez_compressed(path, **payload)
    deployed = SpikingSelector(path)
    with torch.no_grad():
        test_scores = net(torch.from_numpy(np.clip((test_raw-mean)/std, -5, 5))).numpy()
    predicted = np.argmax(np.where(test_masks, test_scores, -np.inf), axis=1)
    matrix = np.zeros((4, 4), dtype=int)
    np.add.at(matrix, (test_labels, predicted), 1)
    # Evaluate the actual exported NumPy runtime against the training model.
    exported_scores = np.array([deployed.predict(row)[0] for row in test_raw[:100]])
    export_agreement = float(np.mean(np.argmax(exported_scores, 1) == np.argmax(test_scores[:100], 1)))
    metrics = dict(seed=seed, train_samples=samples, validation_samples=len(val_raw),
                   test_samples=len(test_raw), class_counts=class_counts.tolist(),
                   best_validation_accuracy=best_accuracy,
                   test_accuracy_masked=float(np.mean(predicted == test_labels)),
                   test_accuracy_unmasked=float(np.mean(np.argmax(test_scores, 1) == test_labels)),
                   confusion_matrix=matrix.tolist(), numpy_export_argmax_agreement=export_agreement,
                   numpy_export_max_score_error=float(np.max(np.abs(exported_scores-test_scores[:100]))),
                   elapsed_seconds=time.perf_counter()-started, epochs=curve)
    (output / 'training.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    manifest = dict(schema_version=1, model_type='two-layer LIF spiking neural network',
                    architecture=[12, 64, 32, 4], temporal_steps=16, beta=0.9, threshold=1,
                    reset='immediate subtractive reset', encoding='constant normalized analog current',
                    decoding='linear readout of mean layer-2 spike counts',
                    state_reset='membranes reset to zero on each skill-selection call',
                    training='supervised geometric-teacher imitation; no RL',
                    features=FEATURE_NAMES, skills=SKILL_NAMES,
                    data_seeds=dict(train=seed, validation=seed+10000, test=seed+20000),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    pytorch_version=torch.__version__, numpy_version=np.__version__,
                    observation_source='privileged simulator obstacle state',
                    validation='synthetic classification only; see evaluation reports for MuJoCo results')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k != 'epochs'}, indent=2), flush=True)
    if export_agreement < 0.99:
        raise RuntimeError('Export/runtime prediction mismatch')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_MODEL.parent)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.samples < 100 or args.epochs < 1:
        parser.error('Use at least 100 samples and one epoch')
    train(args.output, args.samples, args.epochs, args.seed)


if __name__ == '__main__':
    main()
