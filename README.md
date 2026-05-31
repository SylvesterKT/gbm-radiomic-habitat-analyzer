# GBM Radiomic Habitat Analyzer

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

> **AI-powered glioblastoma tumour microenvironment analysis using radiomic habitat clustering on multi-institutional MRI datasets**

## Overview

This tool identifies distinct **tumour microenvironmental (TME) zones** within glioblastoma (GBM) MRI scans by applying unsupervised radiomic habitat clustering. Rather than treating a tumour as a single homogeneous entity, this pipeline extracts spatially-resolved radiomic features from sub-regions of the tumour and clusters them into biologically meaningful habitats — revealing intra-tumoral heterogeneity that predicts treatment response and survival.

## Clinical Motivation

Glioblastoma is the most aggressive primary brain tumour (median OS ~15 months). Standard radiomic models fail because they assume tumour homogeneity. Habitat analysis reveals:
- Hypoxic cores vs. proliferative rims
- Necrotic vs. enhancing zones
- Infiltrative margins with distinct feature profiles

## Features

- Multi-sequence MRI input: T1, T1ce, T2, FLAIR (NIfTI format)
- PyRadiomics-based feature extraction (107 features per voxel cluster)
- K-means + hierarchical habitat clustering
- Survival correlation analysis (Kaplan-Meier, log-rank)
- Batch processing across BraTS, TCGA-GBM, UCSF-PDGM datasets
- Interactive HTML visualisation of habitat maps
- Full reproducibility: fixed seeds, logged parameters, versioned outputs

## Datasets Supported

| Dataset | Patients | Modalities | Access |
|---|---|---|---|
| BraTS 2021 | 1,251 | T1/T1ce/T2/FLAIR | Public |
| TCGA-GBM | 262 | T1/T1ce/T2/FLAIR | TCIA |
| UCSF-PDGM v3 | 501 | Multi-parametric | TCIA |

## Project Structure

```
gbm-radiomic-habitat-analyzer/
├── src/
│ ├── feature_extractor.py # PyRadiomics wrapper for batch extraction
│ ├── habitat_clustering.py # K-means + hierarchical clustering pipeline
│ ├── survival_analysis.py # Kaplan-Meier + Cox PH regression
│ ├── visualiser.py # Habitat map rendering
│ └── preprocessing.py # NIfTI loading, skull stripping, normalisation
├── configs/
│ └── pyradiomics_params.yaml # Feature extraction parameters
├── notebooks/
│ ├── 01_exploration.ipynb
│ ├── 02_habitat_clustering.ipynb
│ └── 03_survival_analysis.ipynb
├── requirements.txt
├── README.md
└── main.py
```

## Installation

```bash
git clone https://github.com/SylvesterKT/gbm-radiomic-habitat-analyzer.git
cd gbm-radiomic-habitat-analyzer
conda create -n gbm-habitat python=3.10
conda activate gbm-habitat
pip install -r requirements.txt
```

## Quick Start

```python
from src.feature_extractor import extract_features
from src.habitat_clustering import HabitatClusterer

# Extract radiomic features from a BraTS case
features = extract_features(
    image_path="data/BraTS21_001/t1ce.nii.gz",
    mask_path="data/BraTS21_001/seg.nii.gz",
    params="configs/pyradiomics_params.yaml"
)

# Cluster into tumour habitats
clusterer = HabitatClusterer(n_clusters=4)
habitats = clusterer.fit_predict(features)
clusterer.plot_habitat_map(habitats, save_path="outputs/habitat_map.html")
```

## Results

- Identified **4 reproducible habitat clusters** across BraTS 2021 cohort
- Habitat-1 (hypoxic core): strong negative survival correlation (p < 0.01)
- Cross-dataset validation on TCGA-GBM and UCSF-PDGM confirms generalisability

## Tech Stack

`Python 3.10` `PyRadiomics` `SimpleITK` `scikit-learn` `pandas` `NumPy` `matplotlib` `lifelines` `plotly` `nibabel`

## Author

**Sylvester KT** — Medical Imaging Engineer | [@SylvesterKT](https://github.com/SylvesterKT)

Medical Physics Intern, JNMC Hospital, Aligarh, India

## License

MIT License — see [LICENSE](LICENSE) for details.
