"""Forecast the next 10-minute appliance energy interval using only known readings."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
from torch import nn

SHA = "2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a"
COLUMNS = ("Appliances", "T1", "RH_1", "T_out", "lights")


def load(path: Path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA:
        raise ValueError("Unexpected UCI archive digest")
    with ZipFile(path) as z, z.open("energydata_complete.csv") as stream:
        rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8")))
    dates = np.array([r["date"] for r in rows], dtype="datetime64[m]")
    if not np.all(np.diff(dates) == np.timedelta64(10, "m")):
        raise ValueError("Irregular timestamps")
    arr = np.asarray([[float(r[c]) for c in COLUMNS] for r in rows], dtype="float32")
    if not np.isfinite(arr).all():
        raise ValueError("Nonfinite input")
    return arr


def windows(arr, lo, hi, width):
    first = max(lo, width)
    if first >= hi or hi > len(arr) or width < 1:
        raise ValueError("Empty or invalid window range")
    return np.stack([arr[i - width : i] for i in range(first, hi)]), arr[first:hi, 0]


class Forecaster(nn.Module):
    def __init__(self, kind, features):
        super().__init__()
        self.kind = kind
        self.core = (
            nn.LSTM(features, 16, batch_first=True)
            if kind == "lstm"
            else nn.Sequential(
                nn.Conv1d(features, 16, 3), nn.ReLU(), nn.Conv1d(16, 16, 3, dilation=2), nn.ReLU()
            )
        )
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        if self.kind == "lstm":
            state = self.core(x)[0][:, -1]
        else:
            state = self.core(x.transpose(1, 2))[:, :, -1]
        return self.head(state).squeeze(-1)


def fit(arr, epochs=4):
    torch.manual_seed(701)
    torch.set_num_threads(2)
    n = len(arr)
    a, b = int(n * 0.70), int(n * 0.85)
    mu = arr[:a].mean(0)
    scale = np.maximum(arr[:a].std(0), 1e-6)
    normal = (arr - mu) / scale
    datasets = [
        tuple(map(torch.from_numpy, windows(normal, start, stop, 12)))
        for start, stop in [(0, a), (a, b), (b, n)]
    ]
    train, cal, test = datasets

    def metrics(cal_hat, hat):
        errors = (hat - test[1]).abs() * scale[0]
        cal_errors = ((cal_hat - cal[1]).abs() * scale[0]).detach().numpy()
        rank = min(len(cal_errors) - 1, int(np.ceil(0.90 * (len(cal_errors) + 1))) - 1)
        radius = float(np.sort(cal_errors)[rank])
        return dict(
            mae_Wh=float(errors.mean()),
            calibration_radius_Wh=radius,
            empirical_interval_coverage=float((errors <= radius).float().mean()),
        )

    results = {"persistence": metrics(cal[0][:, -1, 0], test[0][:, -1, 0])}
    for kind in ("lstm", "tcn"):
        model = Forecaster(kind, len(COLUMNS))
        opt = torch.optim.AdamW(model.parameters(), lr=0.003)
        for _ in range(epochs):
            for ids in torch.randperm(len(train[1])).split(128):
                loss = nn.functional.smooth_l1_loss(model(train[0][ids]), train[1][ids])
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            results[kind] = metrics(model(cal[0]), model(test[0]))
    return dict(
        seed=701,
        epochs=epochs,
        horizon_minutes=10,
        context_minutes=120,
        archive_sha256=SHA,
        train_targets=len(train[1]),
        calibration_targets=len(cal[1]),
        test_targets=len(test[1]),
        results=results,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("report", type=Path)
    args = p.parse_args()
    args.report.write_text(json.dumps(fit(load(args.archive)), indent=2) + "\n")
