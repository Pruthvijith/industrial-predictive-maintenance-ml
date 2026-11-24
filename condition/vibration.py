"""Synthetic rotor study. Group shifts vary shaft speed, harmonics and sensor noise."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn


def simulate(groups=24):
    t = np.arange(512) / 512
    rng = np.random.default_rng(810)
    features = []
    images = []
    labels = []
    assets = []
    for asset in range(groups):
        shaft = rng.uniform(18, 27)
        gain = rng.uniform(0.7, 1.5)
        for state in range(3):
            for _ in range(8):
                phase = rng.uniform(0, 2 * np.pi)
                angle = 2 * np.pi * shaft * t + phase
                x = gain * (np.sin(angle) + (0.15 if state != 1 else 0.65) * np.sin(2 * angle))
                x += rng.normal(0, 0.20, len(t))
                if state == 2:
                    x += (np.sin(angle * 3) > 0.88) * np.sin(2 * np.pi * 145 * t)
                freq, psd = signal.periodogram(x, fs=512)

                def band(lo, hi):
                    return psd[(freq >= lo) & (freq <= hi)].sum()

                features.append(
                    [
                        np.sqrt(np.mean(x * x)),
                        np.max(np.abs(x)) / np.sqrt(np.mean(x * x)),
                        band(1.8 * shaft, 2.2 * shaft) / max(band(0.8 * shaft, 1.2 * shaft), 1e-9),
                        band(100, 220),
                    ]
                )
                _, _, spec = signal.stft(x, fs=512, nperseg=64, noverlap=32, boundary=None, padded=False)
                images.append(np.log1p(100 * np.abs(spec)))
                labels.append(state)
                assets.append(asset)
    return tuple(map(np.asarray, (features, images, labels, assets)))


def experiment():
    torch.manual_seed(810)
    torch.set_num_threads(2)
    f, x, y, assets = simulate()
    mask = assets < 18
    base = make_pipeline(StandardScaler(), LogisticRegression()).fit(f[mask], y[mask])
    x = torch.tensor(x[:, None], dtype=torch.float32)
    labels = torch.tensor(y)
    net = nn.Sequential(
        nn.Conv2d(1, 8, 3), nn.ReLU(), nn.AvgPool2d(2), nn.Flatten(), nn.Linear(8 * 15 * 6, 3)
    )
    opt = torch.optim.AdamW(net.parameters(), lr=0.002)
    for _ in range(10):
        for ids in torch.randperm(int(mask.sum())).split(32):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(net(x[ids]), labels[ids])
            loss.backward()
            opt.step()
    with torch.no_grad():
        cnn = net(x[~mask]).argmax(1).numpy()
    preds = {"spectral_logistic": base.predict(f[~mask]), "spectrogram_cnn": cnn}
    return dict(
        seed=810,
        train_assets=18,
        test_assets=6,
        n=len(y[~mask]),
        sampling_Hz=512,
        results={
            k: dict(
                macro_f1=f1_score(y[~mask], v, average="macro"),
                confusion=confusion_matrix(y[~mask], v).tolist(),
            )
            for k, v in preds.items()
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("report", type=Path)
    args = p.parse_args()
    args.report.write_text(json.dumps(experiment(), indent=2) + "\n")
