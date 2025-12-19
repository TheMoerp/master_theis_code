# BraTS – Unsupervised Anomaly Detection (3D)

Dieses Repository enthält mehrere Pipelines für unüberwachtes Anomalie-Detektion auf dem BraTS-Datensatz (Hirntumore) auf Patch-Basis in 3D. Implementiert sind u. a. Autoencoder (AE), VQ-VAE, f-AnoGAN (GAN), Diffusion (DDPM) sowie feature-basierte Verfahren (Anatomix+KNN, VQ-VAE-Encoder+KNN).

## Installation

```bash
git clone git@github.com:TheMoerp/master-thesis.git
cd master-thesis

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

git clone git clone https://github.com/neel-dey/anatomix.git
pip install -e anatomix

mkdir -p datasets
# Lade/entpacke den gewünschten BraTS2025-GLI-PRE-Challenge-TrainingData-Ordner hier hinein
```

## Datensatz

Standardmäßig wird der BraTS-Datensatz relativ zum Projekt-Root erwartet:

```
./datasets/BraTS2025-GLI-PRE-Challenge-TrainingData/
  ├─ <SubjectID_1>/
  │    ├─ *-t1c.nii.gz
  │    ├─ *-seg.nii.gz (oder *seg.nii.gz)
  │    └─ ...
  ├─ <SubjectID_2>/
  └─ ...
```

### 1) Autoencoder (AE)

Script: `src/ae_brats.py`

```bash
python src/ae_brats.py \
  --num_subjects 1600 \
  --num_epochs 300 \
  --latent_dim 256 \
  --anomaly_labels 1 4 \
```

### 2) VQ-VAE

Script: `src/vqvae_brats.py`

```bash
python src/vqvae_brats.py \
  --num_subjects 1600 \
  --num_epochs 300 \
  --embedding_dim 128 \
  --codebook_size 512 \
  --anomaly_labels 1 4 \
```

Ergebnisse u. a.: `best_vqvae_3d.pth`, `evaluation_results.txt`, Trainingskurven, PR-/ROC-Kurven etc.

### 3) VQ-VAE Encoder + KNN

Script: `src/vqvae_encoder_knn_brats.py`

```bash
python src/vqvae_encoder_knn_brats.py \
  --num_subjects 1600 \
  --embedding_dim 128 \
  --k_neighbors 7 \
  --threshold_percentile 95 \
  --pretrained_weights path/to/best_vqvae_3d.pth \
  --anomaly_labels 1 4 \
```
### 4) Anatomix + KNN

Script: `src/anatomixknn_brats.py`

```bash
python src/anatomixknn_brats.py \
  --num_subjects 1600 \
  --k_neighbors 7 \
  --threshold_percentile 95 \
  --channel_selection_mode unsupervised \
  --anomaly_labels 1 4 \
```

### 5) f-AnoGAN (GAN)

Script: `src/gan_brats.py`

```bash
python src/gan_brats.py \
  --num_subjects 1600 \
  --latent_dim 128 \
  --g_lr 2e-4 \
  --d_lr 2e-4 \
  --e_lr 1e-4 \
  --gan_epochs 80 \
  --encoder_epochs 40 \
  --anomaly_labels 1 4 \
```

## Tipps

- Für schnelle Tests die Anzahl `--num_subjects` reduzieren.