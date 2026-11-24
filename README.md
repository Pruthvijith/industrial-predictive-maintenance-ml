# Condition monitoring and energy forecasting

Compare engineering features and small neural models on two time-series problems. One uses measured appliance energy. The other uses synthetic vibration signals with known harmonic signatures.

Independent, AI-assisted portfolio research milestone · v0.1.0 · Python 3.12 · CPU reference environment.

## Measured outcome

On 2,961 future ten-minute targets, mean absolute error was 26.74 Wh for persistence, 25.57 Wh for LSTM and 25.24 Wh for TCN. A chronological calibration block produced a TCN interval radius of 47.55 Wh and 89.6% empirical coverage. On 144 simulated windows across six unseen assets, spectral logistic regression achieved macro F1 1.000 versus 0.972 for the CNN. A simple frequency feature suits this simulator.

## Architecture and data flow

A pinned UCI archive supplies ten-minute observations. The energy pipeline checks sampling cadence, partitions time, fits scaling on training data and constructs past-only windows. It compares persistence, an LSTM and a temporal convolutional network. A distinct rotor simulator generates shaft speed, harmonics and impulsive oscillation. A logistic baseline uses shaft-order power bands, RMS and crest factor. A CNN consumes short-time Fourier transform magnitudes.

## Dataset and model usage contract

Energy is measured in a single low-energy building. Luis Candanedo, Appliances Energy Prediction, UCI Machine Learning Repository, 2017, DOI 10.24432/C5VC8G, CC BY 4.0. Source archive checksum is checked before reading. Rotor traces are original synthetic signals from 24 virtual assets sampled at 512 Hz. Real public thermal images, industrial plant data and remaining-life labels are outstanding.

The supported purpose is offline research and human-reviewed engineering decision support. Aviation design approval, operational aircraft decisions and regulated certification are outside the validated scope.

## Reproduce

Run from this checkout on Linux x86-64. The reference CPU was an AMD EPYC 9V74. The portfolio index includes a downloader for the pinned models and UCI input.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
python -m pip install -c constraints-cpu.txt -r requirements-dev.txt
python -m condition.vibration work-vibration.json
python -m condition.energy "$ENERGY_ZIP" work-energy.json
```

## Testing and deployment

```bash
python -m pytest --cov=condition --cov-report=term-missing
python -m ruff check .
python -m mypy condition
```

Download the cited UCI archive using the index staging tool or the official URL. Set ENERGY_ZIP to its local path. Run the two modules as offline batch jobs. No production sensor connectivity or on-device inference service was measured.

## Operations and next validation

Reject changes in archive digest or timestamp cadence before training. If a neural model loses to persistence or frequency features, inspect seasonal changes, calibration residuals and per-state confusion before increasing the network size. New acquisition sites require independent evaluation.

See [decisions and interview guide](docs/review.md) and the measured files in [reports](reports/). Hosted CI and production qualification require their own evidence. For dependencies and known advisories see the portfolio index security report.

## Provenance and licensing

Original code and original fixtures use Apache-2.0. Upstream models, data and third-party executables retain their own licences. External weights and source archives are fetched separately. References follow.

- https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
- https://doi.org/10.24432/C5VC8G
- https://creativecommons.org/licenses/by/4.0/

## Two supporting maintenance experiments

Thermal classification uses background-normalised synthetic temperature fields from 30 assets. Under an eight-asset holdout, the physical-feature baseline reached macro F1 0.961 and the CNN reached 0.953 across 128 frames. Neither result establishes field-camera performance. The spare-parts experiment retrieves likely components from synthetic orders. On the last 320 orders, Recall@5 was 19.4% for global popularity, 90.9% for conditional counts and 90.3% for learned embeddings. Neither neural method beat its strongest simple baseline.

```bash
python -m condition.thermography work-thermal.json
python -m condition.parts work-parts.json
```

## Local verification

5 tests passed. Statement coverage of the condition package was 89.03%. Ruff and mypy passed. See [verification record](reports/verification.json).
