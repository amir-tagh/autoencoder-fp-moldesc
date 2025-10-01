# autoencoder-fp-moldesc
Autoencoder Pretraining with Fingerprints and Mordred Descriptors

This repository provides a scalable pipeline for pretraining autoencoders on large molecular datasets, combining Morgan fingerprints and Mordred descriptors stored in efficient memory-mapped arrays (memmaps).

The workflow is designed for datasets with millions of SMILES and runs efficiently on HPC clusters with multiprocessing.

Features

Fingerprint preprocessing
Builds memory-mapped Morgan fingerprints (uint8) for large SMILES datasets.

Mordred descriptor preprocessing
Two-pass computation of Mordred (2D) descriptors with streaming, imputation, and standardization.

Memory-mapped dataset loading
Custom PyTorch IterableDataset (ConcatMemmapIterable) that concatenates fingerprints + descriptors on the fly in shuffled chunks.

Autoencoder training
Simple feedforward autoencoder with configurable latent dimension, trained with MSE loss.

Artifacts saving
Outputs include model weights, training curves, preprocessing metadata, and scikit-learn imputer/scaler objects for downstream use.

Installation

Requirements:

Python ≥ 3.8

RDKit

Mordred

PyTorch

NumPy, Pandas, scikit-learn, joblib

Matplotlib, tqdm

Example installation (conda):

conda create -n ae python=3.9
conda activate ae
conda install -c conda-forge rdkit mordred
pip install torch scikit-learn joblib pandas matplotlib tqdm

Usage

The script has three main commands: build-fp, build-mordred, and train.

1. Build fingerprint memmap
python ae_memmap_fp_mordred.py build-fp \
  --input molecules.csv \
  --out fp_22M.uint8 \
  --fp_bits 1024 \
  --fp_radius 2 \
  --ncpus 32


Input CSV must have a column SMILES.

Produces:

fp_22M.uint8 (memmap)

fp_22M.uint8.meta.json (metadata with nrows, fp_bits, fp_radius)

2. Build Mordred descriptor memmap

Two-pass process:

Pass A computes statistics and saves them to .npz.

Pass B builds the standardized memmap.

# Pass A only (stats)
python ae_memmap_fp_mordred.py build-mordred \
  --input molecules.csv \
  --out mordred_5M.f32 \
  --stats_npz mordred_stats.npz \
  --passA_only

# Full (Pass A + Pass B)
python ae_memmap_fp_mordred.py build-mordred \
  --input molecules.csv \
  --out mordred_5M.f32 \
  --stats_npz mordred_stats.npz \
  --ncpus 32


Outputs:

mordred_5M.f32 (float32 memmap)

mordred_5M.f32.meta.json (metadata with nrows, mordred_dim)

mordred_5M_kept_stats.npz (kept feature names, mean, std)

3. Train autoencoder
python ae_memmap_fp_mordred.py train \
  --fp_memmap fp_22M.uint8 \
  --mordred_memmap mordred_5M.f32 \
  --nrows 5000000 \
  --fp_bits 1024 \
  --fp_radius 2 \
  --latent_dim 128 \
  --epochs 40 \
  --batch_size 4096 \
  --chunk_rows 100000 \
  --num_workers 8 \
  --ncpus 32 \
  --outdir AE_OUT


Outputs in AE_OUT/:

autoencoder_model.pt – trained model weights

training_loss_plot.png – training vs validation loss curves

training_metrics.csv – per-epoch metrics

feature_names.json – combined feature list

preproc.json – preprocessing metadata

imputer.pkl, scaler.pkl – scikit-learn objects for inference

Example Workflow
# Build memmaps
python ae_memmap_fp_mordred.py build-fp \
  --input smiles.csv --out fp_22M.uint8 --fp_bits 1024 --fp_radius 2 --ncpus 32

python ae_memmap_fp_mordred.py build-mordred \
  --input smiles.csv --out mordred_5M.f32 --stats_npz mordred_stats.npz --ncpus 32

# Train model
python ae_memmap_fp_mordred.py train \
  --fp_memmap fp_22M.uint8 --mordred_memmap mordred_5M.f32 \
  --nrows 5000000 --fp_bits 1024 --fp_radius 2 \
  --latent_dim 256 --epochs 50 --batch_size 4096 \
  --outdir AE_OUT

Notes

Ensure nrows matches the smaller of FP and Mordred memmaps (e.g. 5M if Mordred was only computed for 5M molecules).

Large runs should be staged to node-local scratch for performance, then synced back to shared storage.

Supports multi-CPU parallelism (--ncpus, --num_workers).

License

MIT License (adapt as needed).
