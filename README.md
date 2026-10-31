# INT8-Quantized CNN vs. Spiking Neural Network for ECG Arrhythmia Detection

**Author:** A S M Sayem, Dept. of CSE, East West University, Dhaka, Bangladesh
**Focus:** Energy-efficient neural inference for edge cardiac monitoring (hardware-oriented ML)

---

## Motivation

Wearable and implantable cardiac monitors are battery-limited, so a useful classifier has to be accurate, small, *and* cheap to run. Two paradigms are usually proposed for this:

1. **Quantization** of a conventional CNN (reduce numerical precision).
2. **Spiking neural networks (SNNs)**, which communicate through sparse binary events and map naturally onto event-driven neuromorphic hardware.

This project runs a controlled, reproducible comparison of both against an FP32 baseline on the same task, and asks a more specific hardware question: **how few inference timesteps can an SNN use before accuracy collapses, and what does that buy in estimated energy?**

---

## What was done

| Model | Description |
|---|---|
| **FP32 CNN** | Compact 1D-CNN (2 conv blocks + 2 FC layers, 97K params), class-weighted cross-entropy, early stopping |
| **INT8 CNN** | FX graph-mode static post-training quantization (Conv1d + Linear), calibrated on the validation split |
| **SNN** | Same topology with Leaky Integrate-and-Fire neurons (snnTorch, β = 0.9), fast-sigmoid surrogate gradient, trained with BPTT at T = 20, constant-current input encoding |

**Task:** binary classification (normal vs. arrhythmia) on the MIT-BIH Arrhythmia Database (187-sample beat segments).
**Evaluation:** accuracy, precision/recall/F1 (arrhythmia class), ROC-AUC, parameter count, on-disk size, measured CPU latency, and an operation-count energy estimate. All reported numbers are computed in the notebook, nothing is hand-entered.

---

## Results (held-out test set, 21,892 beats)

| Model | Accuracy | F1 (arrhythmia) | ROC-AUC | Params | Size |
|---|---|---|---|---|---|
| FP32 CNN | 97.73% | 0.936 | 0.993 | 97,122 | 385 KB |
| INT8 CNN (static) | 97.69% | 0.935 | 0.993 | ~2.7K stored tensors* | 109 KB |
| SNN (T = 20) | 96.10% | 0.888 | 0.983 | 5,666 | 28 KB |

\*Count reflects stored state-dict tensors of the quantized model, not trainable parameters.

**Key observations**

- **INT8 quantization is nearly free:** −0.04 pp accuracy for a **71.7% size reduction**.
- **The SNN is the smallest model (28 KB)** but gives up ~1.6 pp accuracy relative to the FP32 CNN.
- **CPU latency favors the CNN** (0.19 ms vs. 21 ms per sample). This is expected: SNN inference on a conventional CPU loops over timesteps and cannot exploit event-driven sparsity. It says nothing about neuromorphic hardware.

### Timestep sweep: the accuracy-energy trade-off

The SNN trained at T = 20 was evaluated at smaller T *without retraining*:

| T | Accuracy | Est. energy (pJ) |
|---|---|---|
| 5 | 92.2% | 482,815 |
| **8** | **94.8%** | **773,278** |
| 10 | 95.4% | 967,075 |
| 15 | 96.1% | 1,451,037 |
| 20 | 96.1% | 1,935,169 |
| 30 | 96.0% | 2,903,284 |

Accuracy stays within 2% of the T = 20 result down to **T = 8, with ~60% lower estimated energy**. At T = 8 the SNN's estimated energy (≈0.77 M pJ) falls below the FP32 CNN estimate (≈1.63 M pJ). At T = 20 it does **not**, because the first conv layer receives constant analog input and performs a dense MAC every timestep. Shortening the temporal window is therefore where the SNN's energy advantage actually comes from.

Layer-wise average firing rates were low (conv1 5.8%, conv2 2.2%), which is what makes the spike-driven accumulate (AC) operations cheap in the estimate.

---

## Energy estimation method (and its limits)

Energy is **estimated, not measured.** It is derived from operation counts and per-operation figures from Horowitz (ISSCC 2014, 45 nm):

- FP32 MAC ≈ 4.6 pJ, INT8 MAC ≈ 0.23 pJ, AC (spike event) ≈ 0.03 pJ
- SNN conv1 treated as dense FP32 MACs every timestep; conv2/fc1 as spike-driven ACs scaled by measured firing rates

Memory access, data movement, and routing are **not** modeled, so these are order-of-magnitude *relative* indicators only.

> **Known issue:** the per-layer INT8 detection in the energy cell currently fails to recognize the quantized layers, so the INT8 CNN energy figure in the notebook output equals the FP32 value. It should be disregarded until that check is corrected. The FP32 and SNN estimates and the T-sweep are unaffected.

---

## Limitations

- **Beat-level split, not patient-level.** The released CSVs contain no record IDs, so train/test beats may come from the same patients and generalization is likely overstated.
- Four arrhythmia sub-types are collapsed into one class.
- Energy figures are op-count estimates (see above); no hardware power measurement.
- The T-sweep is zero-shot temporal truncation of a model trained at T = 20, not per-T training.
- Single seed, single run per model; no confidence intervals.
- Spike-raster analysis is a single-example illustration, not a class-level claim.

---

## Future work

1. Patient-level (record-wise) train/test partitioning.
2. Multi-class arrhythmia classification.
3. Quantization-aware training and per-T SNN fine-tuning.
4. Alternative spike encodings (rate / latency coding) to remove the dense first layer.
5. Hardware-in-the-loop power measurement (FPGA / neuromorphic target) to validate the estimates; a natural bridge to ASIC/VLSI-level implementation of the quantized and spiking datapaths.

---

## Reproducing

```bash
pip install torch snntorch numpy pandas scikit-learn matplotlib
```

1. Place `mitbih_train.csv` and `mitbih_test.csv` (Kaggle MIT-BIH heartbeat release) in the data folder referenced in the notebook (default: Google Drive `ecg_project/`).
2. Open `bioedge.ipynb` in Google Colab (GPU recommended) and run all cells.
3. Outputs are written to `figures/` (Figs 1-7) and `artifacts/` (tables, model files). Seeds are fixed (42).

## References

1. Moody & Mark (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE EMB Magazine*, 20(3).
2. Horowitz (2014). Computing's energy problem (and what we can do about it). *ISSCC*.
3. Eshraghian et al. (2021). Training Spiking Neural Networks Using Lessons From Deep Learning. arXiv:2109.12894.
4. Paszke et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*.