"""Retrieve spare parts for simulated maintenance work orders using a two-tower model."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def work_orders(seed=188):
    rng = np.random.default_rng(seed)
    # Asset family and observed fault influence part relevance. The catalogue has 48 parts.
    pairs = []
    for day in range(90):
        for _ in range(16):
            family, fault = map(int, rng.integers([0, 0], [4, 4]))
            if rng.random() < 0.12:
                part = int(rng.integers(48))
            else:
                part = (family * 12 + fault * 3 + int(rng.choice(3, p=[0.65, 0.25, 0.10]))) % 48
            pairs.append((day, family * 4 + fault, part))
    return np.asarray(pairs)


def measure():
    torch.manual_seed(188)
    torch.set_num_threads(1)
    rows = work_orders()
    train, test = rows[rows[:, 0] < 70], rows[rows[:, 0] >= 70]
    query, item = nn.Embedding(16, 8), nn.Embedding(48, 8)
    opt = torch.optim.Adam([*query.parameters(), *item.parameters()], lr=0.02)
    for _ in range(20):
        for ix in torch.randperm(len(train)).split(64):
            q = torch.from_numpy(train[ix, 1])
            p = torch.from_numpy(train[ix, 2])
            opt.zero_grad()
            loss = nn.functional.cross_entropy(query(q) @ item.weight.T, p)
            loss.backward()
            opt.step()
    with torch.no_grad():
        ranked = (query(torch.from_numpy(test[:, 1])) @ item.weight.T).topk(5).indices.numpy()
    popular = np.argsort(-np.bincount(train[:, 2], minlength=48), kind="stable")[:5]
    # A conditional count baseline tests whether embeddings add evidence beyond a frequency table.
    counts = np.ones((16, 48))
    np.add.at(counts, (train[:, 1], train[:, 2]), 1)
    conditional = np.argsort(-counts, axis=1, kind="stable")[test[:, 1], :5]
    return dict(
        train_orders=len(train),
        test_orders=len(test),
        seed=188,
        recall_at_5={
            key: float(np.any(r == test[:, 2, None], axis=1).mean())
            for key, r in [
                ("global_popularity", popular[None, :]),
                ("conditional_counts", conditional),
                ("embeddings", ranked),
            ]
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.write_text(json.dumps(measure(), indent=2) + "\n")
