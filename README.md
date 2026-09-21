# Addressing Representation and Aggregation Biases in Streaming Anomaly Detection through Domain Adaptation and Data Shift Detection

## Overview

This repository contains the official implementation of the scientific paper:

**"Addressing Representation and Aggregation Biases in Streaming Anomaly Detection through Domain Adaptation and Data Shift Detection"**

The proposed framework introduces **Drift Aware Streaming Anomaly Detection (DASAD)** for streaming anomaly detection under **data shift**, **representation bias**, and **aggregation bias**.

The framework combines:

- 🔄 **Streaming Drift Detection**
- 🧠 **Domain-Adversarial Neural Networks (DANN)**
- 📊 **Imbalanced Streaming Learning**
- ⚡ **Online Model Adaptation**

This enables robust anomaly detection in **non‑stationary streaming environments**.

`engine.source_size` defines a fixed labeled prefix (3000 samples in the
example configuration). Scaler S0 and the initial encoder/anomaly classifier
M0 are fitted only on this prefix. Later stream labels are retained exclusively
for evaluation.

After a drift, DASAD collects a raw, unlabeled target window and fits a new
domain-specific scaler only on that observed window. The resulting model and
scaler are activated together (M1+S1, M2+S2, ...). Samples in the collection
window are still predicted by the previous model/scaler pair. The drift
detector always receives values transformed by the fixed source scaler S0 so
that changing preprocessing does not invalidate its historical reference.

The online protocol is prequential: each sample is predicted by the currently
active model before it is passed to the drift detector or an unlabeled target
buffer. After a drift, the previous model continues to predict while the target
batch is collected. DANN adaptation then uses the fixed labeled source and the
unlabeled target batch; it never receives target anomaly labels.

You can also follow the steps presented in jupyter notebook `DASAD_notebook.ipynb`

---

## Repository Structure

```
├── DASAD/
│   ├── engine.py           # Streaming execution engine
│   ├── models.py           # DANN predictor implementation
│   ├── networks.py         # Neural network builders
│   ├── detectors.py        # Drift detection module
│   └── utils.py            # Utility functions
│
├── configs/
│   └── config_ds1.yaml     # Example experiment configuration
│
├── data/
│   └── (dataset files)
│
├── main.py                 # Main training & evaluation script
├── DASAD_notebook.ipynb    # Example notebook
├── requirements.txt
└── README.md
```

---

## Requirements

- Python **3.11.7**
- pip


## Installation

Clone the repository:

```bash
git clone https://github.com/nataliawojak/DASAD.git
cd DASAD
```

Create virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate         # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run experiment using configuration file:

```bash
python main.py --config configs/config_ds1.yaml
```

Example output:

```
F1 score: 0.842
G-mean: 0.791
```
---


## Method Overview


```
Stream Data
     ↓
Drift Detector
     ↓
Domain Adaptation (DANN)
     ↓
Updated Predictor
     ↓
Streaming Predictions
```

## Reproducibility

To reproduce experiments:

1. Prepare dataset
2. Configure YAML file
3. Run main script
4. Compare results

---



