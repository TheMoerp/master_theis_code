#!/usr/bin/env python3

import os
import argparse
import random
import warnings
import time
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.figure_factory as ff

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve, auc as sk_auc
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from common.brats_preprocessing import BraTSPreprocessor, create_unique_results_dir

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class Config:
    def __init__(self):
        # Dataset
        self.dataset_path = "datasets/BraTS2025-GLI-PRE-Challenge-TrainingData"
        self.output_dir = "gan_brats_results"

        # Patch extraction
        self.patch_size = 32
        self.patches_per_volume = 50
        self.min_non_zero_ratio = 0.2
        self.max_normal_to_anomaly_ratio = 3
        self.min_tumor_ratio_in_patch = 0.05

        # Patch quality
        self.min_patch_std = 0.01
        self.min_patch_mean = 0.05
        self.max_tumor_ratio_normal = 0.01
        self.min_tumor_ratio_anomaly = 0.05
        self.max_normal_patches_per_subject = 100
        self.max_anomaly_patches_per_subject = 50

        # Segmentation labels
        self.anomaly_labels = [1, 2, 4]

        # Brain tissue quality
        self.min_brain_tissue_ratio = 0.3
        self.max_background_intensity = 0.1
        self.min_brain_mean_intensity = 0.1
        self.max_high_intensity_ratio = 0.7
        self.high_intensity_threshold = 0.9
        self.edge_margin = 8

        # GAN/Encoder model params
        self.latent_dim = 128
        self.g_learning_rate = 2e-4
        self.d_learning_rate = 2e-4
        self.e_learning_rate = 1e-4
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.batch_size = 8
        self.gan_epochs = 80
        self.encoder_epochs = 40
        self.num_workers = 4 if torch.cuda.is_available() else 0

        # Anomaly score 
        self.alpha_residual = 0.9

        # Anomaly scoring mode fixed to classic f-AnoGAN reconstruction
        self.anomaly_mode = 'reconstruction'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.slice_axis = 'axial'

        self.verbose = False


class BraTSPatchDataset(Dataset):
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


class Generator3D(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        self.unflatten = nn.Unflatten(1, (256, 4, 4, 4))
        self.net = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = self.unflatten(x)
        x = self.net(x)
        return x


class Discriminator3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(256 * 2 * 2 * 2, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.features(x)
        logits = self.classifier(self.flatten(feats))
        if return_features:
            return logits, feats
        return logits


class Encoder3D(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 2 * 2 * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        z = self.fc(self.flatten(h))
        return z


class GenerativeAnomalyDetector:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.anomaly_mode = config.anomaly_mode
        self.G = Generator3D(config.latent_dim).to(self.device)
        self.D = Discriminator3D().to(self.device)
        self.E = Encoder3D(config.latent_dim).to(self.device)

    def train_gan(self, train_loader: DataLoader):
        criterion = nn.BCEWithLogitsLoss()
        optimizerD = optim.Adam(self.D.parameters(), lr=self.config.d_learning_rate, betas=(self.config.beta1, self.config.beta2))
        optimizerG = optim.Adam(self.G.parameters(), lr=self.config.g_learning_rate, betas=(self.config.beta1, self.config.beta2))
        scaler = GradScaler()
        self.G.train()
        self.D.train()
        fixed_noise = torch.randn(self.config.batch_size, self.config.latent_dim, device=self.device)
        total_steps = self.config.gan_epochs * max(1, len(train_loader))
        with tqdm(total=total_steps, desc="GAN Training") as pbar:
            for epoch in range(self.config.gan_epochs):
                for real, _ in train_loader:
                    real = real.to(self.device)
                    bsz = real.size(0)
                    valid = torch.ones((bsz, 1), device=self.device)
                    fake = torch.zeros((bsz, 1), device=self.device)

                    # Train Discriminator
                    optimizerD.zero_grad(set_to_none=True)
                    z = torch.randn(bsz, self.config.latent_dim, device=self.device)
                    with autocast():
                        fake_imgs = self.G(z).detach()
                        real_logits = self.D(real)
                        fake_logits = self.D(fake_imgs)
                        d_real_loss = criterion(real_logits, valid)
                        d_fake_loss = criterion(fake_logits, fake)
                        d_loss = (d_real_loss + d_fake_loss) / 2
                    scaler.scale(d_loss).backward()
                    scaler.step(optimizerD)

                    # Train Generator
                    optimizerG.zero_grad(set_to_none=True)
                    z = torch.randn(bsz, self.config.latent_dim, device=self.device)
                    with autocast():
                        gen_imgs = self.G(z)
                        gen_logits = self.D(gen_imgs)
                        g_loss = criterion(gen_logits, valid)
                    scaler.scale(g_loss).backward()
                    scaler.step(optimizerG)
                    scaler.update()

                    pbar.update(1)
                    pbar.set_postfix({
                        'epoch': f"{epoch+1}/{self.config.gan_epochs}",
                        'd_loss': f"{d_loss.item():.4f}",
                        'g_loss': f"{g_loss.item():.4f}"
                    })

        # Save GAN weights
        torch.save(self.G.state_dict(), os.path.join(self.config.output_dir, 'best_gan_generator.pth'))
        torch.save(self.D.state_dict(), os.path.join(self.config.output_dir, 'best_gan_discriminator.pth'))

    def train_encoder(self, train_loader: DataLoader):
        # Freeze G and D, train E to reconstruct x via G(E(x)) and match D features
        for p in self.G.parameters():
            p.requires_grad = False
        for p in self.D.parameters():
            p.requires_grad = False
        self.G.eval()
        self.D.eval()
        self.E.train()

        optimizerE = optim.Adam(self.E.parameters(), lr=self.config.e_learning_rate, betas=(self.config.beta1, self.config.beta2))
        scaler = GradScaler()
        total_steps = self.config.encoder_epochs * max(1, len(train_loader))
        alpha = self.config.alpha_residual

        with tqdm(total=total_steps, desc="Encoder Training") as pbar:
            for epoch in range(self.config.encoder_epochs):
                for real, _ in train_loader:
                    real = real.to(self.device)
                    optimizerE.zero_grad(set_to_none=True)
                    with autocast():
                        z = self.E(real)
                        recon = self.G(z)
                        # izi Loss
                        residual = torch.mean((real - recon) ** 2, dim=(1, 2, 3, 4))
                        residual_loss = residual.mean()
                        # D Loss
                        _, feats_real = self.D(real, return_features=True)
                        _, feats_fake = self.D(recon, return_features=True)
                        feat_loss = F.l1_loss(feats_fake, feats_real)
                        loss = alpha * residual_loss + (1 - alpha) * feat_loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizerE)
                    scaler.update()
                    pbar.update(1)
                    pbar.set_postfix({
                        'epoch': f"{epoch+1}/{self.config.encoder_epochs}",
                        'enc_loss': f"{loss.item():.4f}",
                        'res': f"{residual_loss.item():.4f}",
                        'feat': f"{feat_loss.item():.4f}"
                    })

        torch.save(self.E.state_dict(), os.path.join(self.config.output_dir, 'best_encoder.pth'))

    @torch.no_grad()
    def compute_anomaly_scores(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.G.eval()
        self.D.eval()
        self.E.eval()
        alpha = self.config.alpha_residual
        scores, labels = [], []
        for real, y in tqdm(data_loader, desc="Scoring"):
            real = real.to(self.device)
            z = self.E(real)
            recon = self.G(z)
            residual = torch.mean((real - recon) ** 2, dim=(1, 2, 3, 4))
            _, feats_real = self.D(real, return_features=True)
            _, feats_fake = self.D(recon, return_features=True)
            feat = torch.mean(torch.abs(feats_real - feats_fake), dim=(1, 2, 3, 4))
            score = alpha * residual + (1 - alpha) * feat
            scores.extend(score.detach().cpu().numpy())
            labels.extend(y.cpu().numpy().flatten())
        return np.array(scores), np.array(labels)

    def find_unsupervised_threshold(self, val_loader: DataLoader) -> float:
        scores, labels = self.compute_anomaly_scores(val_loader)
        normal_scores = scores[labels == 0]
        if len(normal_scores) == 0:
            return float(np.percentile(scores, 95)) if len(scores) else 0.0
        thr = float(np.percentile(normal_scores, 95))
        if self.config.verbose:
            print(f"Normal val scores: mean={normal_scores.mean():.6f}, std={normal_scores.std():.6f}, thr={thr:.6f}")
        return thr

    def evaluate(self, test_loader: DataLoader, val_loader: DataLoader) -> Dict:
        # Load checkpoints if present
        g_path = os.path.join(self.config.output_dir, 'best_gan_generator.pth')
        d_path = os.path.join(self.config.output_dir, 'best_gan_discriminator.pth')
        e_path = os.path.join(self.config.output_dir, 'best_encoder.pth')
        if os.path.exists(g_path):
            self.G.load_state_dict(torch.load(g_path, map_location=self.device))
        if os.path.exists(d_path):
            self.D.load_state_dict(torch.load(d_path, map_location=self.device))
        if os.path.exists(e_path):
            self.E.load_state_dict(torch.load(e_path, map_location=self.device))

        optimal_threshold = self.find_unsupervised_threshold(val_loader)
        scores, true_labels = self.compute_anomaly_scores(test_loader)

        latent_features, _ = self.compute_latent_features(test_loader)

        examples = self.sample_reconstruction_examples(test_loader, max_normal=10, max_anomaly=10)
        preds = (scores > optimal_threshold).astype(int)

        try:
            fpr, tpr, _ = roc_curve(true_labels, scores)
            roc_auc = sk_auc(fpr, tpr)
            precision_curve, recall_curve, _ = precision_recall_curve(true_labels, scores)
            average_precision = sk_auc(recall_curve, precision_curve)
        except Exception:
            roc_auc, average_precision = 0.0, 0.0
        accuracy = accuracy_score(true_labels, preds)
        precision = precision_score(true_labels, preds, zero_division=0)
        recall = recall_score(true_labels, preds, zero_division=0)
        f1 = f1_score(true_labels, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) > 0 else 0
        dsc = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        results = {
            'anomaly_mode': self.anomaly_mode,
            'anomaly_scores': scores,
            'true_labels': true_labels,
            'predictions': preds,
            'latent_features': latent_features,
            'optimal_threshold': optimal_threshold,
            'sample_real': examples.get('real'),
            'sample_recon': examples.get('recon'),
            'sample_residual': examples.get('residual'),
            'sample_feat_residual': examples.get('feat_residual'),
            'sample_labels': examples.get('labels'),
            'roc_auc': roc_auc,
            'average_precision': average_precision,
            'mcc': mcc,
            'dsc': dsc,
            'balanced_accuracy': balanced_accuracy,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'fpr': fpr,
            'fnr': fnr,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        }
        return results

    @torch.no_grad()
    def compute_latent_features(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.E.eval()
        latents, labels = [], []
        for real, y in tqdm(data_loader, desc="Latent features"):
            real = real.to(self.device)
            z = self.E(real)
            latents.append(z.detach().cpu().numpy())
            labels.extend(y.cpu().numpy().flatten())
        if len(latents) > 0:
            latents = np.concatenate(latents, axis=0)
        else:
            latents = np.zeros((0, self.config.latent_dim), dtype=np.float32)
        return latents, np.array(labels)

    @torch.no_grad()
    def sample_reconstruction_examples(self, data_loader: DataLoader, max_normal: int = 10, max_anomaly: int = 10) -> Dict[str, np.ndarray]:
        self.G.eval(); self.D.eval(); self.E.eval()
        real_list, recon_list, res_list, feat_res_list, labels_list = [], [], [], [], []
        n_norm, n_anom = 0, 0
        for real, y in data_loader:
            real = real.to(self.device)
            y_np = y.cpu().numpy().flatten()
            z = self.E(real)
            recon = self.G(z)
            residual = torch.abs(real - recon)

            _, feats_real = self.D(real, return_features=True)
            _, feats_fake = self.D(recon, return_features=True)
            feat_res = torch.mean(torch.abs(feats_real - feats_fake), dim=1, keepdim=True)  # [B,1,d1,d2,d3]
            feat_res_up = F.interpolate(feat_res, size=real.shape[2:], mode='trilinear', align_corners=False)

            for i in range(real.size(0)):
                label = int(y_np[i])
                if label == 0 and n_norm < max_normal:
                    real_list.append(real[i:i+1].cpu().numpy())
                    recon_list.append(recon[i:i+1].cpu().numpy())
                    res_list.append(residual[i:i+1].cpu().numpy())
                    feat_res_list.append(feat_res_up[i:i+1].cpu().numpy())
                    labels_list.append(label)
                    n_norm += 1
                elif label == 1 and n_anom < max_anomaly:
                    real_list.append(real[i:i+1].cpu().numpy())
                    recon_list.append(recon[i:i+1].cpu().numpy())
                    res_list.append(residual[i:i+1].cpu().numpy())
                    feat_res_list.append(feat_res_up[i:i+1].cpu().numpy())
                    labels_list.append(label)
                    n_anom += 1
                if n_norm >= max_normal and n_anom >= max_anomaly:
                    break
            if n_norm >= max_normal and n_anom >= max_anomaly:
                break
        if len(real_list) == 0:
            return {'real': None, 'recon': None, 'residual': None, 'feat_residual': None, 'labels': None}
        real_arr = np.concatenate(real_list, axis=0)
        recon_arr = np.concatenate(recon_list, axis=0)
        res_arr = np.concatenate(res_list, axis=0)
        feat_res_arr = np.concatenate(feat_res_list, axis=0)
        labels_arr = np.array(labels_list)
        return {'real': real_arr, 'recon': recon_arr, 'residual': res_arr, 'feat_residual': feat_res_arr, 'labels': labels_arr}

class Visualizer:
    def __init__(self, config: Config):
        self.config = config

    def plot_confusion_matrix(self, true_labels: np.ndarray, predictions: np.ndarray):
        cm = confusion_matrix(true_labels, predictions)
        if cm.shape != (2, 2):
            if self.config.verbose:
                print("WARNING: Confusion matrix is not 2x2. Skipping plot.")
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
        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Blues', font_colors=['black', 'white'])
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
        except ValueError as e:
            print(f"ERROR: Could not save confusion matrix plot: {e}")

    def plot_roc_curve(self, true_labels: np.ndarray, scores: np.ndarray):
        fpr, tpr, _ = roc_curve(true_labels, scores)
        auc = sk_auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {auc:.2f})")
        plt.fill_between(fpr, tpr, step="pre", alpha=0.25, color="#aec7e8")
        plt.plot([0, 1], [0, 1], linestyle="--", color="#888888", lw=1, label="Chance")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
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
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        plt.savefig(os.path.join(self.config.output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_score_histogram(self, scores: np.ndarray, true_labels: np.ndarray, optimal_threshold: float):
        normal_scores = scores[true_labels == 0]
        anomaly_scores = scores[true_labels == 1]
        plt.figure(figsize=(12, 6))
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        plt.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {optimal_threshold:.6f}')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'anomaly_score_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_report(self, results: Dict):
        self.plot_confusion_matrix(results['true_labels'], results['predictions'])
        self.plot_roc_curve(results['true_labels'], results['anomaly_scores'])
        self.plot_precision_recall_curve(results['true_labels'], results['anomaly_scores'])
        self.plot_score_histogram(results['anomaly_scores'], results['true_labels'], results['optimal_threshold'])

        if 'latent_features' in results and len(results['latent_features']) > 0:
            self.plot_latent_space_visualization(results['latent_features'], results['true_labels'])

        if results.get('sample_real') is not None:
            self.save_residual_examples(results['sample_real'], results['sample_recon'], results['sample_residual'], results['sample_labels'])
            if results.get('sample_feat_residual') is not None:
                self.save_feature_residual_examples(results['sample_real'], results['sample_recon'], results['sample_feat_residual'], results['sample_labels'])

        self.save_real_vs_generated_grid()
        print(f"\nAll visualizations saved to: {self.config.output_dir}")

    def plot_latent_space_visualization(self, latent_features: np.ndarray, true_labels: np.ndarray):
        print("Creating latent space visualizations (PCA & t-SNE) from encoder embeddings...")
        # besser lesbarerr tsne
        if len(latent_features) > 2000:
            idx = np.random.choice(len(latent_features), 2000, replace=False)
            latent = latent_features[idx]
            labels = true_labels[idx]
        else:
            latent = latent_features
            labels = true_labels

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # PCA
        pca = PCA(n_components=2)
        pca_feat = pca.fit_transform(latent)
        ax1.scatter(pca_feat[labels == 0, 0], pca_feat[labels == 0, 1], c='blue', alpha=0.6, label='Normal', s=20)
        ax1.scatter(pca_feat[labels == 1, 0], pca_feat[labels == 1, 1], c='red', alpha=0.6, label='Anomaly', s=20)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax1.set_title('PCA of Encoder Latent Features')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(latent)//4)))
        tsne_feat = tsne.fit_transform(latent)
        ax2.scatter(tsne_feat[labels == 0, 0], tsne_feat[labels == 0, 1], c='blue', alpha=0.6, label='Normal', s=20)
        ax2.scatter(tsne_feat[labels == 1, 0], tsne_feat[labels == 1, 1], c='red', alpha=0.6, label='Anomaly', s=20)
        ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2')
        ax2.set_title('t-SNE of Encoder Latent Features')
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'latent_space_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_patch_triplet(self, ax_row, vol_real, vol_recon, vol_heat, title_heat: str, axis='axial'):

        if axis == 'axial':
            axes = 2
        elif axis == 'coronal':
            axes = 1
        else:
            axes = 0
        for j, frac in enumerate([0.35, 0.5, 0.65]):
            s = max(0, min(vol_real.shape[axes]-1, int(vol_real.shape[axes] * frac)))
            if axes == 2:
                r_slice = vol_real[0, 0, :, :, s]
                x_slice = vol_recon[0, 0, :, :, s]
                h_slice = vol_heat[0, 0, :, :, s]
            elif axes == 1:
                r_slice = vol_real[0, 0, :, s, :]
                x_slice = vol_recon[0, 0, :, s, :]
                h_slice = vol_heat[0, 0, :, s, :]
            else:
                r_slice = vol_real[0, 0, s, :, :]
                x_slice = vol_recon[0, 0, s, :, :]
                h_slice = vol_heat[0, 0, s, :, :]
            ax_row[0, j].imshow(r_slice, cmap='gray'); ax_row[0, j].set_title('Input'); ax_row[0, j].axis('off')
            ax_row[1, j].imshow(x_slice, cmap='gray'); ax_row[1, j].set_title('Recon'); ax_row[1, j].axis('off')
            im = ax_row[2, j].imshow(h_slice, cmap='inferno'); ax_row[2, j].set_title(title_heat); ax_row[2, j].axis('off')

    def save_residual_examples(self, real: np.ndarray, recon: np.ndarray, residual: np.ndarray, labels: np.ndarray, axis: str = 'axial'):
        out_dir = os.path.join(self.config.output_dir, 'qualitative_examples')
        os.makedirs(out_dir, exist_ok=True)
        for i in range(real.shape[0]):
            fig, axes = plt.subplots(3, 3, figsize=(10, 10))
            self._save_patch_triplet(axes, real[i:i+1], recon[i:i+1], residual[i:i+1], 'Residual', axis)
            tag = 'normal' if labels[i] == 0 else 'anomaly'
            fig.suptitle(f'{tag.capitalize()} example #{i+1}', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{tag}_residual_{i+1:03d}.png'), dpi=200, bbox_inches='tight')
            plt.close()

    def save_feature_residual_examples(self, real: np.ndarray, recon: np.ndarray, feat_residual: np.ndarray, labels: np.ndarray, axis: str = 'axial'):
        out_dir = os.path.join(self.config.output_dir, 'qualitative_examples')
        os.makedirs(out_dir, exist_ok=True)
        for i in range(real.shape[0]):
            fig, axes = plt.subplots(3, 3, figsize=(10, 10))
            self._save_patch_triplet(axes, real[i:i+1], recon[i:i+1], feat_residual[i:i+1], 'Feat-Residual', axis)
            tag = 'normal' if labels[i] == 0 else 'anomaly'
            fig.suptitle(f'{tag.capitalize()} example (feature) #{i+1}', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{tag}_feat_residual_{i+1:03d}.png'), dpi=200, bbox_inches='tight')
            plt.close()

    def save_real_vs_generated_grid(self, n: int = 16):
        # später machen
        pass


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='3D f-AnoGAN Anomaly Detection for BraTS')
    parser.add_argument('--num_subjects', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--patches_per_volume', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--g_lr', type=float, default=2e-4)
    parser.add_argument('--d_lr', type=float, default=2e-4)
    parser.add_argument('--e_lr', type=float, default=1e-4)
    parser.add_argument('--gan_epochs', type=int, default=80)
    parser.add_argument('--encoder_epochs', type=int, default=40)
    parser.add_argument('--alpha_residual', type=float, default=0.9)
    parser.add_argument('--output_dir', type=str, default='gan_brats_results')
    parser.add_argument('--dataset_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData')
    parser.add_argument('--max_normal_to_anomaly_ratio', type=int, default=3)
    parser.add_argument('--min_tumor_ratio_in_patch', type=float, default=0.05)
    parser.add_argument('--anomaly_labels', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    config = Config()
    config.num_subjects = args.num_subjects
    config.patch_size = args.patch_size
    config.patches_per_volume = args.patches_per_volume
    config.batch_size = args.batch_size
    config.latent_dim = args.latent_dim
    config.g_learning_rate = args.g_lr
    config.d_learning_rate = args.d_lr
    config.e_learning_rate = args.e_lr
    config.gan_epochs = args.gan_epochs
    config.encoder_epochs = args.encoder_epochs
    config.alpha_residual = args.alpha_residual
    config.output_dir = create_unique_results_dir('gan_brats')
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir) if os.path.basename(_script_dir) == 'src' else _script_dir
    config.dataset_path = args.dataset_path if os.path.isabs(args.dataset_path) else os.path.join(_project_root, args.dataset_path)
    config.max_normal_to_anomaly_ratio = args.max_normal_to_anomaly_ratio
    config.min_tumor_ratio_in_patch = args.min_tumor_ratio_in_patch
    config.anomaly_labels = args.anomaly_labels
    config.verbose = args.verbose
    config.anomaly_mode = 'reconstruction'

    label_names = {0: "Background/Normal", 1: "NCR/NET (Necrotic/Non-enhancing)", 2: "ED (Edema)", 4: "ET (Enhancing Tumor)"}
    anomaly_names = [f"{label} ({label_names.get(label, 'Unknown')})" for label in config.anomaly_labels] if config.verbose else [f"{l}" for l in config.anomaly_labels]
    if config.verbose:
        print("="*60)
        print("3D f-AnoGAN ANOMALY DETECTION FOR BRATS")
        print("="*60)
        print(f"Device: {config.device}")
        print(f"Dataset path: {config.dataset_path}")
        print(f"Output directory: {config.output_dir}")
        print(f"Patch size: {config.patch_size}^3  | Batch size: {config.batch_size}")
        print(f"GAN epochs: {config.gan_epochs} | Encoder epochs: {config.encoder_epochs}")
        print(f"Anomaly labels: {anomaly_names}")
        print("="*60)
        print("\n1. Processing dataset and extracting patches...")
    else:
        print(f"3D f-AnoGAN | Anomaly labels: {anomaly_names} | Output: {config.output_dir}")

    processor = BraTSPreprocessor(config)
    patches, labels, subjects = processor.process_dataset(config.num_subjects)
    if len(patches) == 0:
        print("Error: No patches extracted")
        return

    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        print("ERROR: Need both normal and anomalous patches for evaluation.")
        return

    unique_subjects = list(set(subjects))
    np.random.shuffle(unique_subjects)
    n_subjects = len(unique_subjects)
    train_subjects = unique_subjects[:int(0.6 * n_subjects)]
    val_subjects = unique_subjects[int(0.6 * n_subjects):int(0.8 * n_subjects)]
    test_subjects = unique_subjects[int(0.8 * n_subjects):]

    train_indices = [i for i, s in enumerate(subjects) if s in train_subjects]
    val_indices = [i for i, s in enumerate(subjects) if s in val_subjects]
    test_indices = [i for i, s in enumerate(subjects) if s in test_subjects]

    X_train_all = patches[train_indices]
    y_train_all = labels[train_indices]
    X_val_all = patches[val_indices]
    y_val_all = labels[val_indices]
    X_test = patches[test_indices]
    y_test = labels[test_indices]

    train_normal_mask = (y_train_all == 0)
    val_normal_mask = (y_val_all == 0)
    X_train_normal = X_train_all[train_normal_mask]
    y_train_normal = y_train_all[train_normal_mask]
    X_val_normal = X_val_all[val_normal_mask]
    y_val_normal = y_val_all[val_normal_mask]

    if config.verbose:
        print(f"\n=== SUBJECT-LEVEL SPLIT ===")
        print(f"Train: {len(X_train_normal)} from {len(train_subjects)} subjects")
        print(f"Val:   {len(X_val_normal)} from {len(val_subjects)} subjects")
        print(f"Test:        {len(X_test)} from {len(test_subjects)} subjects")
        print("="*60)

    assert len(set(train_subjects) & set(val_subjects)) == 0
    assert len(set(train_subjects) & set(test_subjects)) == 0
    assert len(set(val_subjects) & set(test_subjects)) == 0

    train_dataset = BraTSPatchDataset(X_train_normal, y_train_normal)
    val_dataset = BraTSPatchDataset(X_val_normal, y_val_normal)
    test_dataset = BraTSPatchDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    if config.verbose:
        print("\n2. Training GAN...")
    detector = GenerativeAnomalyDetector(config)
    detector.train_gan(train_loader)

    if config.verbose:
        print("\n3. Training encoder")
    detector.train_encoder(train_loader)

    if config.verbose:
        print("\n4. Evaluating on test set...")
    results = detector.evaluate(test_loader, val_loader)

    visualizer = Visualizer(config)
    visualizer.create_summary_report(results)

    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    time_parts = []
    if hours > 0:
        time_parts.append(f"{hours}h")
    if minutes > 0:
        time_parts.append(f"{minutes}m")
    time_parts.append(f"{seconds}s")
    time_formatted = " ".join(time_parts)

    results_file = os.path.join(config.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("3D f-AnoGAN Anomaly Detection Results\n")
        f.write("="*60 + "\n")
        f.write(f"ROC AUC:           {results['roc_auc']:.4f}\n")
        f.write(f"Average Precision: {results['average_precision']:.4f}\n")
        f.write(f"Matthews Correlation: {results['mcc']:.4f}\n")
        f.write(f"Dice Similarity Coeff: {results['dsc']:.4f}\n")
        f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
        f.write(f"F1 Score:          {results['f1_score']:.4f}\n")
        f.write(f"Precision:         {results['precision']:.4f}\n")
        f.write(f"Recall:            {results['recall']:.4f}\n")
        f.write(f"Specificity:       {results['specificity']:.4f}\n")
        f.write(f"Accuracy:          {results['accuracy']:.4f}\n")
        f.write(f"Threshold Used:    {results['optimal_threshold']:.6f}\n")
        f.write("="*60 + "\n")
        f.write("Configuration:\n")
        f.write(f"  Patch size: {config.patch_size}\n")
        f.write(f"  Patches per volume: {config.patches_per_volume}\n")
        f.write(f"  Latent dimension: {config.latent_dim}\n")
        f.write(f"  G LR: {config.g_learning_rate}\n")
        f.write(f"  D LR: {config.d_learning_rate}\n")
        f.write(f"  E LR: {config.e_learning_rate}\n")
        f.write(f"  Batch size: {config.batch_size}\n")
        f.write(f"  GAN epochs: {config.gan_epochs}\n")
        f.write(f"  Encoder epochs: {config.encoder_epochs}\n")
        f.write(f"  Anomaly mode: {config.anomaly_mode}\n")
        f.write(f"  Training samples: {len(X_train_normal)}\n")
        f.write(f"  Validation samples: {len(X_val_normal)}\n")
        f.write(f"  Test samples: {len(X_test)}\n")
        f.write(f"  Training subjects: {len(train_subjects)}\n")
        f.write(f"  Validation subjects: {len(val_subjects)}\n")
        f.write(f"  Test subjects: {len(test_subjects)}\n")
        f.write(f"  Anomaly labels used: {config.anomaly_labels}\n")
        f.write(f"  Anomaly label descriptions: {anomaly_names}\n")
        f.write("="*60 + "\n")
        f.write("EXECUTION TIME:\n")
        f.write(f"  Total time: {time_formatted} ({total_time:.1f} seconds)\n")
        print(f"\nPipeline completed in {time_formatted}. Results saved to: {results_file}")


if __name__ == "__main__":
    main()




