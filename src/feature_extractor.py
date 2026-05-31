"""
feature_extractor.py
PyRadiomics-based radiomic feature extraction pipeline for GBM MRI analysis.
Author: Sylvester KT (@SylvesterKT)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

logger = logging.getLogger(__name__)


def load_nifti(path: Union[str, Path]) -> sitk.Image:
    """Load a NIfTI image using SimpleITK."""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    image = sitk.ReadImage(path)
    logger.info(f"Loaded image: {path} | Size: {image.GetSize()}")
    return image


def extract_features(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    params: Union[str, Path] = "configs/pyradiomics_params.yaml",
    label: int = 1,
) -> pd.Series:
    """
    Extract radiomic features from a single MRI + mask pair.

    Args:
        image_path: Path to NIfTI image (.nii or .nii.gz)
        mask_path:  Path to NIfTI segmentation mask
        params:     Path to PyRadiomics parameter YAML file
        label:      Mask label value to extract features for

    Returns:
        pd.Series of radiomic features with feature names as index
    """
    extractor = featureextractor.RadiomicsFeatureExtractor(str(params))
    extractor.settings["label"] = label

    result = extractor.execute(str(image_path), str(mask_path))

    # Filter to numeric features only
    features = {
        k: float(v)
        for k, v in result.items()
        if k.startswith(("original_", "wavelet_", "log_"))
    }

    logger.info(f"Extracted {len(features)} features from {Path(image_path).name}")
    return pd.Series(features, name=Path(image_path).parent.name)


def batch_extract(
    data_dir: Union[str, Path],
    image_name: str = "t1ce.nii.gz",
    mask_name: str = "seg.nii.gz",
    params: Union[str, Path] = "configs/pyradiomics_params.yaml",
    label: int = 1,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Batch extract features from a directory of BraTS-style cases.

    Directory structure expected:
        data_dir/
            patient_001/
                t1ce.nii.gz
                seg.nii.gz
            patient_002/
                ...

    Args:
        data_dir:   Root directory containing patient subdirectories
        image_name: Filename of the MRI image within each patient folder
        mask_name:  Filename of the segmentation mask
        params:     PyRadiomics parameter YAML file path
        label:      Segmentation label to extract
        n_jobs:     Number of parallel jobs (1 = sequential)

    Returns:
        pd.DataFrame with patients as rows and features as columns
    """
    data_dir = Path(data_dir)
    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if not patient_dirs:
        raise ValueError(f"No patient directories found in {data_dir}")

    logger.info(f"Found {len(patient_dirs)} patients in {data_dir}")
    records: List[pd.Series] = []

    for patient_dir in patient_dirs:
        image_path = patient_dir / image_name
        mask_path = patient_dir / mask_name

        if not image_path.exists() or not mask_path.exists():
            logger.warning(f"Skipping {patient_dir.name}: missing files")
            continue

        try:
            features = extract_features(image_path, mask_path, params, label)
            records.append(features)
        except Exception as e:
            logger.error(f"Failed for {patient_dir.name}: {e}")
            continue

    df = pd.DataFrame(records)
    logger.info(f"Extraction complete: {df.shape[0]} patients, {df.shape[1]} features")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch radiomic feature extraction")
    parser.add_argument("--data_dir", required=True, help="Path to patient data directory")
    parser.add_argument("--output", default="features.csv", help="Output CSV path")
    parser.add_argument("--params", default="configs/pyradiomics_params.yaml")
    parser.add_argument("--image", default="t1ce.nii.gz")
    parser.add_argument("--mask", default="seg.nii.gz")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    df = batch_extract(args.data_dir, args.image, args.mask, args.params)
    df.to_csv(args.output)
    print(f"Saved {df.shape[0]} x {df.shape[1]} feature matrix to {args.output}")
