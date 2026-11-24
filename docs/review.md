# Design decision and review guide

## Decision record 001

Chronological validation prevents future loads from entering training. Test prediction windows may contain historically observed readings from before the split, as at inference time. Fixed residual quantiles describe empirical uncertainty only. Single-building autocorrelation limits any coverage guarantee. Equipment-separated rotor testing guards against sharing windows from one asset between sets, though train and test still share a simulator.

## Operational response

Reject changes in archive digest or timestamp cadence before training. If a neural model loses to persistence or frequency features, inspect seasonal changes, calibration residuals and per-state confusion before increasing the network size. New acquisition sites require independent evaluation.

## Interview practice

Use these answers after reproducing the experiment. Describe the work as an AI-assisted independent portfolio project.

### Why is persistence hard to beat?

Adjacent ten-minute loads are correlated. Predicting the last reading is a realistic operational benchmark. The TCN reduced MAE by about 1.49 Wh on this particular test block.

### What does a high harmonic-to-fundamental ratio mean here?

The simulated second state increases the second shaft-order component. Scaling bands by shaft speed keeps the feature aligned as RPM varies. Real fault causes would require independent engineering labels.

### How would you establish readiness?

Add new buildings and independently measured machine conditions, assess drift over seasons, and measure the cost of false alarms and missed faults.

