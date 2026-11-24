"""Segment controlled synthetic thermal fields and compare a CNN against physical features."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fields():
    rng = np.random.default_rng(821)
    gy, gx = np.mgrid[:24, :24]
    maps = []
    labels = []
    assets = []
    for group in range(30):
        ambient = rng.uniform(18, 35)
        center = rng.uniform(8, 16, 2)
        for fault in (0, 1):
            for _ in range(8):
                x, y = center + rng.normal(0, 0.8, 2)
                peak = rng.normal(12 if fault else 5, 2)
                field = (
                    ambient
                    + peak * np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / 20)
                    + rng.normal(0, 0.45, gx.shape)
                )
                maps.append(field)
                labels.append(fault)
                assets.append(group)
    return np.asarray(maps, dtype=np.float32), np.asarray(labels), np.asarray(assets)


def experiment():
    torch.manual_seed(821)
    torch.set_num_threads(2)
    data, y, assets = fields()
    # Subtract each field's own background. Absolute ambient varies across equipment.
    norm = data - np.median(data, axis=(1, 2), keepdims=True)
    f = np.column_stack((np.percentile(norm, 99, axis=(1, 2)), (norm > 6).mean(axis=(1, 2))))
    train = assets < 22
    baseline = make_pipeline(StandardScaler(), LogisticRegression()).fit(f[train], y[train])
    x = torch.from_numpy(norm[:, None] / 10)
    labels = torch.from_numpy(y)
    net = nn.Sequential(nn.Conv2d(1, 8, 3), nn.ReLU(), nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(8, 2))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(12):
        for ids in torch.randperm(int(train.sum())).split(32):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(net(x[ids]), labels[ids])
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = net(x[~train]).argmax(1).numpy()
    return dict(
        seed=821,
        unit="synthetic degrees Celsius",
        held_out_assets=8,
        n=int((~train).sum()),
        physical_features_f1=f1_score(y[~train], baseline.predict(f[~train]), average="macro"),
        cnn_f1=f1_score(y[~train], pred, average="macro"),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.write_text(json.dumps(experiment(), indent=2) + "\n")
