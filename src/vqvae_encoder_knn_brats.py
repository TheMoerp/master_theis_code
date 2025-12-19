#!/usr/bin/env python3

import os
import argparse
import random
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.figure_factory as ff

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc as sk_auc,
    confusion_matrix,
)
import common.brats_preprocessing as br_prep
from common.brats_preprocessing import BraTSPreprocessor, create_unique_results_dir
from common.metrics import evaluate_binary_classification, threshold_from_percentile
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class Config:
    """Container for dataset paths, model, and kNN parameters."""
    def __init__(self):
        self.dataset_path = "datasets/BraTS2025-GLI-PRE-Challenge-TrainingData"
        self.output_dir = "vqvae_encoder_knn_results"

        self.patch_size = 32
        self.patches_per_volume = 50
        self.min_non_zero_ratio = 0.2
        self.max_normal_to_anomaly_ratio = 3
        self.min_tumor_ratio_in_patch = 0.05

        self.min_patch_std = 0.01
        self.min_patch_mean = 0.05
        self.max_tumor_ratio_normal = 0.01
        self.min_tumor_ratio_anomaly = 0.05
        self.max_normal_patches_per_subject = 100
        self.max_anomaly_patches_per_subject = 50

        self.anomaly_labels = [1, 2, 4]

        self.min_brain_tissue_ratio = 0.3
        self.max_background_intensity = 0.1
        self.min_brain_mean_intensity = 0.1
        self.max_high_intensity_ratio = 0.7
        self.high_intensity_threshold = 0.9
        self.edge_margin = 8

        self.embedding_dim = 128  # channels output by encoder
        self.feature_pooling = "mean" 

        self.k_neighbors = 7
        self.threshold_percentile = 95.0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4 if torch.cuda.is_available() else 0

        self.verbose = False


class BraTSPatchDataset(Dataset):
    """Simple Dataset wrapper for 3D patches."""
    def __init__(self, patches: np.ndarray, labels: np.ndarray, transform=None):
        self.patches = patches
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        patch = torch.FloatTensor(patch).unsqueeze(0)
        label = torch.FloatTensor([label])
        if self.transform:
            patch = self.transform(patch)
        return patch, label


class VectorQuantizer(nn.Module):
    """Standard VQ layer; kept to mirror VQ-VAE checkpoints."""
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        b, c, d, h, w = z.shape
        z_perm = z.permute(0, 2, 3, 4, 1).contiguous()
        z_flat = z_perm.view(-1, c)
        x_sq = torch.sum(z_flat ** 2, dim=1, keepdim=True)
        e_sq = torch.sum(self.embedding.weight ** 2, dim=1)
        xe = torch.matmul(z_flat, self.embedding.weight.t())
        distances = x_sq + e_sq.unsqueeze(0) - 2 * xe
        encoding_indices = torch.argmin(distances, dim=1)
        z_q_flat = self.embedding(encoding_indices)
        z_q = z_q_flat.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        embedding_loss = F.mse_loss(z_q.detach(), z)
        commitment_loss = self.commitment_cost * F.mse_loss(z_q, z.detach())
        vq_loss = embedding_loss + commitment_loss
        z_q = z + (z_q - z).detach()
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(z_flat.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return z_q, vq_loss, perplexity, encoding_indices.view(b, d, h, w)


class VQVAE3D(nn.Module):
    """3D encoder backbone; decoder is omitted because we only use embeddings."""
    def __init__(self, input_channels: int = 1, embedding_dim: int = 256, codebook_size: int = 512,
                 commitment_beta: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(2)

        self.encoder_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool3d(2)

        self.encoder_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool3d(2)

        self.encoder_conv4 = nn.Sequential(
            nn.Conv3d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.quantizer = VectorQuantizer(num_embeddings=codebook_size,
                                         embedding_dim=embedding_dim,
                                         commitment_cost=commitment_beta)
        self.decoder = nn.Identity()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder_conv1(x)
        e1_pool = self.pool1(e1)
        e2 = self.encoder_conv2(e1_pool)
        e2_pool = self.pool2(e2)
        e3 = self.encoder_conv3(e2_pool)
        e3_pool = self.pool3(e3)
        e4 = self.encoder_conv4(e3_pool)
        return e4


def subject_level_split_indices(subjects: List[str], train_frac=0.6, val_frac=0.2):
    """Delegate to shared subject-level split helper."""
    return br_prep.subject_level_split(subjects, train_frac, val_frac)


def fit_knn_on_normal(train_features: np.ndarray, k: int):
    from sklearn.neighbors import NearestNeighbors
    nns = NearestNeighbors(n_neighbors=min(k, len(train_features)), algorithm='auto')
    nns.fit(train_features)
    return nns


def knn_anomaly_scores(nbrs, ref_features: np.ndarray, query_features: np.ndarray, k: int) -> np.ndarray:
    distances, _ = nbrs.kneighbors(query_features, n_neighbors=min(k, len(ref_features)))
    return distances.mean(axis=1)


def threshold_from_normal(val_scores: np.ndarray, percentile: float) -> float:
    return threshold_from_percentile(val_scores, percentile)


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict:
    return evaluate_binary_classification(y_true, scores, threshold)


class Visualizer:
    """Utility plots for evaluating binary anomaly detection."""
    def __init__(self, config: Config):
        self.config = config

    def plot_confusion_matrix(self, true_labels: np.ndarray, predictions: np.ndarray):
        cm = confusion_matrix(true_labels, predictions)
        if cm.shape != (2, 2):
            return
        tn, fp, fn, tp = cm.ravel()
        z = [[tn, fp], [fn, tp]]
        x = ['Normal (0)', 'Anomaly (1)']
        y = ['Normal (0)', 'Anomaly (1)']
        row_sums = cm.sum(axis=1)
        z_text = [
            [f"{tn}<br>({tn/row_sums[0]*100:.1f}%)" if row_sums[0] > 0 else str(tn),
             f"{fp}<br>({fp/row_sums[0]*100:.1f}%)" if row_sums[0] > 0 else str(fp)],
            [f"{fn}<br>({fn/row_sums[1]*100:.1f}%)" if row_sums[1] > 0 else str(fn),
             f"{tp}<br>({tp/row_sums[1]*100:.1f}%)" if row_sums[1] > 0 else str(tp)]
        ]
        fig = ff.create_annotated_heatmap(
            z, x=x, y=y, annotation_text=z_text, colorscale='Blues',
            font_colors=['black', 'white']
        )
        fig.update_layout(
            title_text='<b>Confusion Matrix</b><br>(Count and Percentage)',
            title_x=0.5,
            xaxis=dict(title='<b>Predicted Label</b>'),
            yaxis=dict(title='<b>True Label</b>', autorange='reversed'),
            font=dict(size=14)
        )
        fig.update_xaxes(side="bottom")
        output_path = os.path.join(self.config.output_dir, 'confusion_matrix.png')
        try:
            fig.write_image(output_path, width=800, height=800, scale=2)
        except ValueError:
            pass

    def plot_roc_curve(self, true_labels: np.ndarray, scores: np.ndarray):
        fpr, tpr, _ = roc_curve(true_labels, scores)
        roc_auc_value = sk_auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc_value:.2f})")
        plt.fill_between(fpr, tpr, step="pre", alpha=0.25, color="#aec7e8")
        plt.plot([0, 1], [0, 1], linestyle="--", color="#888888", lw=1, label="Chance")
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        plt.savefig(os.path.join(self.config.output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_precision_recall_curve(self, true_labels: np.ndarray, scores: np.ndarray):
        precision, recall, _ = precision_recall_curve(true_labels, scores)
        pr_auc_value = sk_auc(recall, precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="#ff7f0e", lw=2, label=f"PR (AUC = {pr_auc_value:.2f})")
        plt.fill_between(recall, precision, step="pre", alpha=0.25, color="#ffbb78")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        plt.savefig(os.path.join(self.config.output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_score_histogram(self, scores: np.ndarray, true_labels: np.ndarray, threshold: float):
        normal_scores = scores[true_labels == 0]
        anomaly_scores = scores[true_labels == 1]
        plt.figure(figsize=(12, 6))
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.6f}')
        plt.xlabel('Anomaly Score (kNN distance)')
        plt.ylabel('Density')
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'score_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_latent_space_visualization(self, features: np.ndarray, true_labels: np.ndarray):
        if len(features) == 0 or features.ndim != 2:
            return
        feats = features
        labels = true_labels
        if len(feats) > 2000:
            idx = np.random.choice(len(feats), 2000, replace=False)
            feats = feats[idx]
            labels = labels[idx]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(feats)
        ax1.scatter(pca_features[labels == 0, 0], pca_features[labels == 0, 1], c='blue', alpha=0.6, label='Normal', s=20)
        ax1.scatter(pca_features[labels == 1, 0], pca_features[labels == 1, 1], c='red', alpha=0.6, label='Anomaly', s=20)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title('PCA of Latent Features')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(feats)//4)))
        tsne_features = tsne.fit_transform(feats)
        ax2.scatter(tsne_features[labels == 0, 0], tsne_features[labels == 0, 1], c='blue', alpha=0.6, label='Normal', s=20)
        ax2.scatter(tsne_features[labels == 1, 0], tsne_features[labels == 1, 1], c='red', alpha=0.6, label='Anomaly', s=20)
        ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2'); ax2.set_title('t-SNE of Latent Features')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'latent_space_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()


def extract_encoder_features(model: VQVAE3D, loader: DataLoader, device: torch.device, pooling: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
    """Run encoder forward passes and pool to feature vectors."""
    model.eval()
    feats = []
    labels_all = []
    with torch.no_grad():
        for data, labels in tqdm(loader, desc="Extracting VQVAE encoder features"):
            x = data.to(device)
            ctx = torch.amp.autocast('cuda') if torch.cuda.is_available() else nullcontext()
            with ctx:
                z = model.encode(x)  # [B, C, D, H, W]
            if pooling == "mean_max_std":
                mean = torch.mean(z, dim=(2, 3, 4))
                mx = torch.amax(z, dim=(2, 3, 4))
                sd = torch.std(z, dim=(2, 3, 4))
                f = torch.cat([mean, mx, sd], dim=1)
            else:
                f = torch.mean(z, dim=(2, 3, 4))
            feats.append(f.cpu().numpy())
            labels_all.append(labels.cpu().numpy().flatten())
    features = np.vstack(feats) if len(feats) else np.zeros((0,))
    labels_np = np.concatenate(labels_all) if len(labels_all) else np.zeros((0,))
    return features, labels_np


def save_results(config: Config, results: Dict, y_true: np.ndarray, scores: np.ndarray, threshold: float,
                 train_n: int, val_n: int, test_n: int,
                 train_subjects_n: int, val_subjects_n: int, test_subjects_n: int):
    out = os.path.join(config.output_dir, 'evaluation_results.txt')
    with open(out, 'w') as f:
        f.write("VQ-VAE encoder + kNN anomaly detection results\n")
        f.write("="*60 + "\n")
        f.write(f"ROC AUC:           {results['roc_auc']:.4f}\n")
        f.write(f"Average Precision: {results['average_precision']:.4f}\n")
        f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
        f.write(f"F1 Score:          {results['f1_score']:.4f}\n")
        f.write(f"Precision:         {results['precision']:.4f}\n")
        f.write(f"Recall:            {results['recall']:.4f}\n")
        f.write(f"Specificity:       {(2*results['balanced_accuracy'] - results['recall']):.4f}\n")
        f.write(f"Accuracy:          {results['accuracy']:.4f}\n")
        f.write(f"MCC:               {results['mcc']:.4f}\n")
        f.write(f"DSC:               {results['dsc']:.4f}\n")
        f.write(f"Threshold Used:    {threshold:.6f}\n")
        f.write("="*60 + "\n")
        f.write("Configuration:\n")
        f.write(f"  Patch size: {config.patch_size}\n")
        f.write(f"  Patches per volume: {config.patches_per_volume}\n")
        f.write(f"  K neighbors: {config.k_neighbors}\n")
        f.write(f"  Threshold percentile: {config.threshold_percentile}\n")
        f.write(f"  Feature pooling: {config.feature_pooling}\n")
        f.write(f"  Training samples: {train_n}\n")
        f.write(f"  Validation samples: {val_n}\n")
        f.write(f"  Test samples: {test_n}\n")
        f.write(f"  Training subjects: {train_subjects_n}\n")
        f.write(f"  Validation subjects: {val_subjects_n}\n")
        f.write(f"  Test subjects: {test_subjects_n}\n")
        f.write(f"  Anomaly labels used: {config.anomaly_labels}\n")
    print(f"Results saved to: {out}")


def main():
    parser = argparse.ArgumentParser(description='VQ-VAE encoder + kNN anomaly detection for BraTS patches')
    parser.add_argument('--num_subjects', type=int, default=None, help='Number of subjects to use (default: all)')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of 3D patches (default: 32)')
    parser.add_argument('--patches_per_volume', type=int, default=50, help='Number of patches per volume (default: 50)')
    parser.add_argument('--output_dir', type=str, default='vqvae_encoder_knn_results', help='Output directory')
    parser.add_argument('--dataset_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData', help='Path to BraTS dataset')
    parser.add_argument('--max_normal_to_anomaly_ratio', type=int, default=3, help='Max ratio of normal to anomaly patches')
    parser.add_argument('--min_tumor_ratio_in_patch', type=float, default=0.05, help='Minimum tumor ratio in anomalous patches')
    parser.add_argument('--anomaly_labels', type=int, nargs='+', default=[1, 2, 4], help='BraTS labels to consider anomalous (default: 1 2 4)')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Encoder output channels (default: 128)')
    parser.add_argument('--k_neighbors', type=int, default=7, help='Number of neighbors for KNN')
    parser.add_argument('--threshold_percentile', type=float, default=95.0, help='Percentile for threshold on validation NORMAL scores (0-100)')
    parser.add_argument('--feature_pooling', type=str, choices=['mean', 'mean_max_std'], default='mean', help='Pooling over encoder feature map')
    parser.add_argument('--pretrained_weights', type=str, required=True, help='Path to VQ-VAE weights (from vqvae_brats.py)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    start_time = time.time()

    cfg = Config()
    cfg.num_subjects = args.num_subjects
    cfg.patch_size = args.patch_size
    cfg.patches_per_volume = args.patches_per_volume
    cfg.output_dir = args.output_dir
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir) if os.path.basename(_script_dir) == 'src' else _script_dir
    cfg.dataset_path = args.dataset_path if os.path.isabs(args.dataset_path) else os.path.join(_project_root, args.dataset_path)
    cfg.max_normal_to_anomaly_ratio = args.max_normal_to_anomaly_ratio
    cfg.min_tumor_ratio_in_patch = args.min_tumor_ratio_in_patch
    cfg.anomaly_labels = args.anomaly_labels
    cfg.embedding_dim = args.embedding_dim
    cfg.k_neighbors = args.k_neighbors
    cfg.threshold_percentile = args.threshold_percentile
    cfg.feature_pooling = args.feature_pooling
    cfg.verbose = args.verbose
    cfg.output_dir = create_unique_results_dir('vqvae_encoder_knn_brats')

    print(f"VQ-VAE-Encoder+KNN | Anomaly labels: {cfg.anomaly_labels} | Output: {cfg.output_dir}")

    # 1) Extract patches
    processor = BraTSPreprocessor(cfg)
    patches, labels, subjects = processor.process_dataset(cfg.num_subjects)
    if len(patches) == 0:
        print("No patches extracted. Exiting.")
        return

    # 2) Subject-level split
    idx_train, idx_val, idx_test, train_subj, val_subj, test_subj = br_prep.subject_level_split(subjects)
    X_train_all, y_train_all = patches[idx_train], labels[idx_train]
    X_val_all, y_val_all = patches[idx_val], labels[idx_val]
    X_test, y_test = patches[idx_test], labels[idx_test]
    train_normal_mask = (y_train_all == 0)
    val_normal_mask = (y_val_all == 0)
    X_train_normal = X_train_all[train_normal_mask]
    X_val_normal = X_val_all[val_normal_mask]
    print(f"Train normal: {len(X_train_normal)} | Val normal: {len(X_val_normal)} | Test total: {len(X_test)}")

    # 3) DataLoaders for encoder feature extraction
    train_dataset = BraTSPatchDataset(X_train_normal, np.zeros(len(X_train_normal)))
    val_dataset = BraTSPatchDataset(X_val_normal, np.zeros(len(X_val_normal)))
    test_dataset = BraTSPatchDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=cfg.num_workers)

    # 4) Initialize VQ-VAE Encoder
    model = VQVAE3D(input_channels=1, embedding_dim=cfg.embedding_dim, codebook_size=512, commitment_beta=0.25).to(cfg.device)
    
    def _load_encoder_weights_or_fail(model_obj, state_dict, checkpoint_path: str):
        encoder_block_prefixes = ('encoder_conv1', 'encoder_conv2', 'encoder_conv3', 'encoder_conv4')
        normalized = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
        model_sd = model_obj.state_dict()
        encoder_keys = [k for k in model_sd.keys() if k.startswith(encoder_block_prefixes)]

        missing = [k for k in encoder_keys if k not in normalized]
        shape_mismatch = [k for k in encoder_keys if k in normalized and normalized[k].shape != model_sd[k].shape]
        if missing or shape_mismatch:
            details = []
            if missing:
                details.append(f"missing keys: {missing}")
            if shape_mismatch:
                details.append(f"shape mismatch: {shape_mismatch}")
            raise ValueError(f"Incompatible encoder weights in '{checkpoint_path}'; " + "; ".join(details))

        updated = {k: normalized[k] for k in encoder_keys}
        model_sd.update(updated)
        model_obj.load_state_dict(model_sd, strict=False)
        print(f"Loaded encoder weights from {checkpoint_path}")

    if not args.pretrained_weights:
        raise ValueError("Argument --pretrained_weights is required.")
    if not os.path.isfile(args.pretrained_weights):
        raise FileNotFoundError(f"Checkpoint '{args.pretrained_weights}' not found.")
    try:
        state = torch.load(args.pretrained_weights, map_location=cfg.device)
        _load_encoder_weights_or_fail(model, state, args.pretrained_weights)
    except Exception as e:
        raise RuntimeError(f"Failed to load required pretrained encoder weights from '{args.pretrained_weights}': {e}") from e

    # 5) Extract encoder features
    train_feats, _ = extract_encoder_features(model, train_loader, cfg.device, pooling=cfg.feature_pooling)
    val_feats, _ = extract_encoder_features(model, val_loader, cfg.device, pooling=cfg.feature_pooling)
    test_feats, test_labels = extract_encoder_features(model, test_loader, cfg.device, pooling=cfg.feature_pooling)

    # 6) Fit KNN on normal train features
    knn = fit_knn_on_normal(train_feats, cfg.k_neighbors)

    # 7) Compute scores
    val_scores = knn_anomaly_scores(knn, train_feats, val_feats, cfg.k_neighbors)
    test_scores = knn_anomaly_scores(knn, train_feats, test_feats, cfg.k_neighbors)

    # 8) Threshold & evaluate
    thr = threshold_from_normal(val_scores, cfg.threshold_percentile)
    results = evaluate_scores(test_labels.astype(int), test_scores, thr)

    # 9) Save results and visualizations
    vis = Visualizer(cfg)
    vis.plot_confusion_matrix(test_labels, results['predictions'])
    vis.plot_roc_curve(test_labels, test_scores)
    vis.plot_precision_recall_curve(test_labels, test_scores)
    vis.plot_score_histogram(test_scores, test_labels, thr)
    vis.plot_latent_space_visualization(test_feats, test_labels)
    save_results(
        cfg, results, test_labels, test_scores, thr,
        train_n=len(X_train_normal), val_n=len(X_val_normal), test_n=len(X_test),
        train_subjects_n=len(train_subj), val_subjects_n=len(val_subj), test_subjects_n=len(test_subj)
    )

    total_time = time.time() - start_time
    print(f"\nPipeline completed in {total_time/60:.1f} min. Results saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()