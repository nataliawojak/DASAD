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
git clone https://github.com/<your-username>/DASAD.git
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

You can also follow the steps presented in jupyter notebook `DASAD_notebook.ipynb`

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





