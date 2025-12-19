#!/usr/bin/env python3

import os
import argparse
import random
import warnings
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
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    confusion_matrix,
    roc_curve, precision_recall_curve, auc as sk_auc,
    accuracy_score, precision_score, recall_score
)
from common.metrics import evaluate_binary_classification, threshold_from_percentile
from common.brats_preprocessing import BraTSPreprocessor, create_unique_results_dir
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class Config:
    def __init__(self):
        # Dataset paths
        self.dataset_path = "datasets/BraTS2025-GLI-PRE-Challenge-TrainingData"
        self.output_dir = "vqvae_brats_results"

        # Patch extraction parameters
        self.patch_size = 32
        self.patches_per_volume = 50
        self.min_non_zero_ratio = 0.2
        self.max_normal_to_anomaly_ratio = 3
        self.min_tumor_ratio_in_patch = 0.05

        # Additional patch quality parameters
        self.min_patch_std = 0.01
        self.min_patch_mean = 0.05
        self.max_tumor_ratio_normal = 0.01
        self.min_tumor_ratio_anomaly = 0.05
        self.max_normal_patches_per_subject = 100
        self.max_anomaly_patches_per_subject = 50

        self.anomaly_labels = [1, 2, 4]

        # Brain tissue quality parameters for normal patches
        self.min_brain_tissue_ratio = 0.3
        self.max_background_intensity = 0.1
        self.min_brain_mean_intensity = 0.1
        self.max_high_intensity_ratio = 0.7
        self.high_intensity_threshold = 0.9
        self.edge_margin = 8

        # Model parameters
        self.learning_rate = 5e-5
        self.batch_size = 8
        self.num_epochs = 100
        self.early_stopping_patience = 20

        # VQ-VAE specific parameters
        self.codebook_size = 512
        self.embedding_dim = 128
        self.commitment_beta = 0.25

        # Training parameters
        self.train_test_split = 0.8
        self.validation_split = 0.2

        # Visualization parameters
        self.slice_axis = 'axial'

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4 if torch.cuda.is_available() else 0


def _format_anomaly_labels(labels: List[int], verbose: bool = False) -> List[str]:
    label_names = {
        0: "Background/Normal",
        1: "NCR/NET (Necrotic/Non-enhancing)",
        2: "ED (Edema)",
        4: "ET (Enhancing Tumor)"
    }
    if verbose:
        return [f"{label} ({label_names.get(label, 'Unknown')})" for label in labels]
    return [f"{label}" for label in labels]


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


class VectorQuantizer(nn.Module):
    """
    Standard VQ layer (no EMA) for 3D feature maps.
    Input: z_e in shape (B, C, D, H, W)
    Output: z_q same shape, vq_loss, perplexity, indices
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        # z: (B, C, D, H, W)
        b, c, d, h, w = z.shape
        # Move channel to last dim to compute distances
        z_perm = z.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        z_flat = z_perm.view(-1, c)  # (B*D*H*W, C)

        # Compute distances to embeddings
        x_sq = torch.sum(z_flat ** 2, dim=1, keepdim=True) 
        e_sq = torch.sum(self.embedding.weight ** 2, dim=1)
        xe = torch.matmul(z_flat, self.embedding.weight.t()) 
        distances = x_sq + e_sq.unsqueeze(0) - 2 * xe 

        encoding_indices = torch.argmin(distances, dim=1)  # (N)
        z_q_flat = self.embedding(encoding_indices)  # (N, C)
        z_q = z_q_flat.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)

        # Losses
        embedding_loss = F.mse_loss(z_q.detach(), z)
        commitment_loss = self.commitment_cost * F.mse_loss(z_q, z.detach())
        vq_loss = embedding_loss + commitment_loss

        z_q = z + (z_q - z).detach()

        # Perplexity
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(z_flat.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, vq_loss, perplexity, encoding_indices.view(b, d, h, w)


class VQVAE3D(nn.Module):
    def __init__(self, input_channels: int = 1, embedding_dim: int = 128, codebook_size: int = 512,
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
        self.pool4 = nn.MaxPool3d(2) 

        self.quantizer = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_beta,
        )

        # Decoder

        self.decoder_conv4 = nn.Sequential(
            nn.Conv3d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # 2->4
        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # 4->8

        self.decoder_conv3 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # 8->16

        self.decoder_conv2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # 16->32

        self.decoder_conv1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder_conv1(x)
        e1_pool = self.pool1(e1)
        e2 = self.encoder_conv2(e1_pool)
        e2_pool = self.pool2(e2)
        e3 = self.encoder_conv3(e2_pool)
        e3_pool = self.pool3(e3)
        e4 = self.encoder_conv4(e3_pool)
        e4_pool = self.pool4(e4)
        return e4_pool

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        z_up = self.upsample4(z_q)
        d4 = self.decoder_conv4(z_up)
        d4_up = self.upsample3(d4)
        d3 = self.decoder_conv3(d4_up)
        d3_up = self.upsample2(d3)
        d2 = self.decoder_conv2(d3_up)
        d2_up = self.upsample1(d2)
        out = self.decoder_conv1(d2_up)
        return out

    def forward(self, x: torch.Tensor):
        z_e = self.encode(x)
        z_q, vq_loss, perplexity, _ = self.quantizer(z_e)
        x_recon = self.decode(z_q)
        latent_features = torch.mean(z_q, dim=(2, 3, 4))
        return x_recon, latent_features, vq_loss, perplexity


class VQVAEAnomalyDetector:
    def __init__(self, config: Config):
        self.config = config
        self.model = VQVAE3D(
            input_channels=1,
            embedding_dim=config.embedding_dim,
            codebook_size=config.codebook_size,
            commitment_beta=config.commitment_beta,
        ).to(config.device)
        self.scaler = GradScaler()

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        scaler = GradScaler()
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        total_steps = self.config.num_epochs * max(1, len(train_loader))
        with tqdm(total=total_steps, desc="Training Progress") as pbar:
            for epoch in range(self.config.num_epochs):
                self.model.train()
                train_loss = 0.0
                normal_samples_processed = 0
                for batch_idx, (data, labels) in enumerate(train_loader):
                    data = data.to(self.config.device)
                    labels = labels.to(self.config.device)
                    normal_mask = (labels == 0).squeeze()
                    if normal_mask.sum() == 0:
                        pbar.update(1)
                        continue
                    normal_data = data[normal_mask]
                    normal_samples_processed += normal_data.size(0)
                    optimizer.zero_grad()
                    with autocast():
                        recon, _, vq_loss, _ = self.model(normal_data)
                        recon_loss = criterion(recon, normal_data)
                        loss = recon_loss + vq_loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss += loss.item()
                    pbar.update(1)
                    pbar.set_postfix({
                        'Epoch': f'{epoch+1}/{self.config.num_epochs}',
                        'Loss': f'{loss.item():.6f}',
                        'Normal_samples': normal_samples_processed
                    })
                avg_train_loss = train_loss / max(1, len(train_loader))
                train_losses.append(avg_train_loss)

                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, labels in val_loader:
                        data = data.to(self.config.device)
                        labels = labels.to(self.config.device)
                        normal_mask = (labels == 0).squeeze()
                        if normal_mask.sum() == 0:
                            continue
                        normal_data = data[normal_mask]
                        with autocast():
                            recon, _, vq_loss, _ = self.model(normal_data)
                            recon_loss = criterion(recon, normal_data)
                            loss = recon_loss + vq_loss
                        val_loss += loss.item()
                avg_val_loss = val_loss / max(1, len(val_loader))
                val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), os.path.join(self.config.output_dir, 'best_vqvae_3d.pth'))
                else:
                    patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    pbar.write(f"Early stopping triggered after {epoch+1} epochs")
                    break
        if getattr(self.config, 'verbose', False):
            print(f"\nTraining completed. Best validation loss: {best_val_loss:.6f}")
        self.save_training_plots(train_losses, val_losses)
        return train_losses, val_losses

    def save_training_plots(self, train_losses: List[float], val_losses: List[float]):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VQ-VAE Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'training_losses.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def calculate_reconstruction_errors(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        reconstruction_errors = []
        true_labels = []
        latent_features = []
        with torch.no_grad():
            for data, labels in tqdm(data_loader, desc="Calculating reconstruction errors"):
                data = data.to(self.config.device)
                with autocast():
                    reconstructed, latent, _, _ = self.model(data)
                mse = torch.mean((data - reconstructed) ** 2, dim=(1, 2, 3, 4))
                reconstruction_errors.extend(mse.cpu().numpy())
                true_labels.extend(labels.cpu().numpy().flatten())
                latent_features.extend(latent.cpu().numpy())
        return np.array(reconstruction_errors), np.array(true_labels), np.array(latent_features)

    def find_unsupervised_threshold(self, val_loader: DataLoader) -> float:
        self.model.eval()
        normal_val_errors = []
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Computing validation errors"):
                data = data.to(self.config.device)
                labels = labels.to(self.config.device)
                normal_mask = (labels == 0).squeeze()
                if normal_mask.sum() == 0:
                    continue
                normal_data = data[normal_mask]
                with autocast():
                    reconstructed, _, _, _ = self.model(normal_data)
                mse = torch.mean((normal_data - reconstructed) ** 2, dim=(1, 2, 3, 4))
                normal_val_errors.extend(mse.cpu().numpy())
        normal_val_errors = np.array(normal_val_errors)
        if getattr(self.config, 'verbose', False):
            print("\nTHRESHOLD DETERMINATION")
            print("=" * 60)
            print(f"Normal validation errors - Count: {len(normal_val_errors)}")
            print(f"Normal validation errors - Mean: {normal_val_errors.mean():.6f}, Std: {normal_val_errors.std():.6f}")
            print(f"Normal validation errors - Min: {normal_val_errors.min():.6f}, Max: {normal_val_errors.max():.6f}")
        threshold = threshold_from_percentile(normal_val_errors, 95)
        if getattr(self.config, 'verbose', False):
            print(f"Selected threshold (95th percentile): {threshold:.6f}")
            print("=" * 60)
        return threshold

    def evaluate(self, test_loader: DataLoader, val_loader: DataLoader) -> Dict:
        model_path = os.path.join(self.config.output_dir, 'best_vqvae_3d.pth')
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.config.device)
                # Load only if checkpoint matches new architecture naming
                if any(k.startswith('encoder_conv1') for k in state_dict.keys()):
                    self.model.load_state_dict(state_dict, strict=True)
                    if getattr(self.config, 'verbose', False):
                        print("Loaded best VQ-VAE model for evaluation")
                else:
                    print("WARNING: Checkpoint incompatible with current architecture. Using in-memory model weights.")
            except Exception as e:
                print(f"WARNING: Failed to load checkpoint due to: {e}. Using in-memory model weights.")
        optimal_threshold = self.find_unsupervised_threshold(val_loader)
        reconstruction_errors, true_labels, latent_features = self.calculate_reconstruction_errors(test_loader)
        print("\nEVALUATION: 3D VQ-VAE Results")
        print("=" * 60)
        print(f"Total test samples: {len(reconstruction_errors)}")
        print(f"Normal samples: {np.sum(true_labels == 0)}")
        print(f"Anomalous samples: {np.sum(true_labels == 1)}")
        eval_res = evaluate_binary_classification(true_labels, reconstruction_errors, optimal_threshold)
        predictions = eval_res['predictions']
        roc_auc = eval_res['roc_auc']
        average_precision = eval_res['average_precision']
        accuracy = eval_res['accuracy']
        precision = eval_res['precision']
        recall = eval_res['recall']
        f1 = eval_res['f1_score']
        balanced_accuracy = eval_res['balanced_accuracy']
        mcc = eval_res['mcc']
        dsc = eval_res['dsc']
        fpr = eval_res['fpr']
        fnr = eval_res['fnr']
        sensitivity = recall
        specificity = 1.0 - fpr
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        normal_errors = reconstruction_errors[true_labels == 0]
        anomaly_errors = reconstruction_errors[true_labels == 1]
        separation_ratio = anomaly_errors.mean() / normal_errors.mean() if normal_errors.mean() > 0 else 0
        print("\nVQ-VAE PERFORMANCE:")
        print("=" * 60)
        print(f"ROC AUC:                  {roc_auc:.4f}")
        print(f"Average Precision:        {average_precision:.4f}")
        print(f"Matthews Correlation:     {mcc:.4f}")
        print(f"Dice Similarity Coeff:    {dsc:.4f}")
        print(f"Balanced Accuracy:        {balanced_accuracy:.4f}")
        print(f"F1 Score:                 {f1:.4f}")
        print(f"Precision:                {precision:.4f}")
        print(f"Recall (Sensitivity):     {recall:.4f}")
        print(f"Specificity:              {specificity:.4f}")
        print(f"Accuracy:                 {accuracy:.4f}")
        print(f"False Positive Rate:      {fpr:.4f}")
        print(f"False Negative Rate:      {fnr:.4f}")
        print(f"Threshold Used:           {optimal_threshold:.6f}")
        print("=" * 60)

        print("\nPOST-HOC RECONSTRUCTION ERROR ANALYSIS:")
        print(f"Normal errors    - Mean: {normal_errors.mean():.6f}, Std: {normal_errors.std():.6f}")
        print(f"Anomaly errors   - Mean: {anomaly_errors.mean():.6f}, Std: {anomaly_errors.std():.6f}")
        print(f"Separation ratio - Anomaly/Normal: {separation_ratio:.3f}")

        print("\nCONFUSION MATRIX ANALYSIS:")
        print(f"True Negatives:      {tn}")
        print(f"False Positives:     {fp}")
        print(f"False Negatives:     {fn}")
        print(f"True Positives:      {tp}")
        results = {
            'reconstruction_errors': reconstruction_errors,
            'true_labels': true_labels,
            'predictions': predictions,
            'latent_features': latent_features,
            'optimal_threshold': optimal_threshold,
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


class Visualizer:
    def __init__(self, config: Config):
        self.config = config

    def plot_confusion_matrix(self, true_labels: np.ndarray, predictions: np.ndarray, eval_metrics: Optional[Dict] = None):
        cm = confusion_matrix(true_labels, predictions)
        if cm.shape != (2, 2):
            if getattr(self.config, 'verbose', False):
                print("WARNING: Confusion matrix is not 2x2. Skipping plot generation.")
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
        total_samples = cm.sum()
        if eval_metrics:
            accuracy = eval_metrics.get('accuracy', 0)
            precision = eval_metrics.get('precision', 0)
            recall = eval_metrics.get('recall', 0)
            specificity = eval_metrics.get('specificity', 0)
        else:
            accuracy = accuracy_score(true_labels, predictions) if total_samples > 0 else 0
            precision = precision_score(true_labels, predictions, zero_division=0) if total_samples > 0 else 0
            recall = recall_score(true_labels, predictions, zero_division=0) if total_samples > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        summary_text = (f'Total Samples: {total_samples}<br>'
                        f'Accuracy: {accuracy:.3f}<br>'
                        f'Precision: {precision:.3f}<br>'
                        f'Recall: {recall:.3f}<br>'
                        f'Specificity: {specificity:.3f}')
        fig.add_annotation(
            text=summary_text,
            align='left',
            showarrow=False,
            xref='paper', yref='paper', x=0.0, y=-0.28,
            bordercolor="black", borderwidth=1, bgcolor="lightgray", font_size=12
        )
        fig.update_layout(margin=dict(t=100, b=150))
        output_path = os.path.join(self.config.output_dir, 'confusion_matrix.png')
        try:
            fig.write_image(output_path, width=800, height=800, scale=2)
        except ValueError as e:
            print(f"ERROR: Could not save confusion matrix plot: {e}")
            print("Please install 'plotly' and 'kaleido' (`pip install plotly kaleido`).")

    def plot_roc_curve(self, true_labels: np.ndarray, reconstruction_errors: np.ndarray):
        fpr, tpr, _ = roc_curve(true_labels, reconstruction_errors)
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

    def plot_precision_recall_curve(self, true_labels: np.ndarray, reconstruction_errors: np.ndarray):
        precision, recall, _ = precision_recall_curve(true_labels, reconstruction_errors)
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

    def plot_reconstruction_error_histogram(self, reconstruction_errors: np.ndarray, true_labels: np.ndarray, optimal_threshold: float):
        normal_errors = reconstruction_errors[true_labels == 0]
        anomaly_errors = reconstruction_errors[true_labels == 1]
        plt.figure(figsize=(12, 6))
        plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        plt.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {optimal_threshold:.6f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'reconstruction_error_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_latent_space_visualization(self, latent_features: np.ndarray, true_labels: np.ndarray):
        print("Creating latent space visualizations...")
        if len(latent_features) > 2000:
            indices = np.random.choice(len(latent_features), 2000, replace=False)
            latent_features = latent_features[indices]
            true_labels = true_labels[indices]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        print("Computing PCA...")
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(latent_features)
        ax1.scatter(pca_features[true_labels == 0, 0], pca_features[true_labels == 0, 1], c='blue', alpha=0.6, label='Normal', s=20)
        ax1.scatter(pca_features[true_labels == 1, 0], pca_features[true_labels == 1, 1], c='red', alpha=0.6, label='Anomaly', s=20)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title('PCA of Latent Features')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_features)//4))
        tsne_features = tsne.fit_transform(latent_features)
        ax2.scatter(tsne_features[true_labels == 0, 0], tsne_features[true_labels == 0, 1], c='blue', alpha=0.6, label='Normal', s=20)
        ax2.scatter(tsne_features[true_labels == 1, 0], tsne_features[true_labels == 1, 1], c='red', alpha=0.6, label='Anomaly', s=20)
        ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2'); ax2.set_title('t-SNE of Latent Features')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'latent_space_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_report(self, results: Dict):
        self.plot_confusion_matrix(results['true_labels'], results['predictions'], eval_metrics=results)
        self.plot_roc_curve(results['true_labels'], results['reconstruction_errors'])
        self.plot_precision_recall_curve(results['true_labels'], results['reconstruction_errors'])
        self.plot_reconstruction_error_histogram(results['reconstruction_errors'], results['true_labels'], results['optimal_threshold'])
        self.plot_latent_space_visualization(results['latent_features'], results['true_labels'])
        print(f"\nAll visualizations saved to: {self.config.output_dir}")


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='3D VQ-VAE Anomaly Detection for BraTS Dataset')
    parser.add_argument('--num_subjects', type=int, default=None, help='Number of subjects to use (default: all)')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of 3D patches (default: 32)')
    parser.add_argument('--patches_per_volume', type=int, default=50, help='Number of patches per volume (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training (default: 8)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate (default: 5e-5)')
    parser.add_argument('--embedding_dim', type=int, default=128, help='VQ-VAE embedding dimension (default: 128)')
    parser.add_argument('--codebook_size', type=int, default=512, help='Number of VQ codebook entries (default: 512)')
    parser.add_argument('--commitment_beta', type=float, default=0.25, help='VQ commitment cost beta (default: 0.25)')
    parser.add_argument('--output_dir', type=str, default='vqvae_brats_results', help='Output directory (default: vqvae_brats_results)')
    parser.add_argument('--dataset_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData', help='Path to BraTS dataset')
    parser.add_argument('--max_normal_to_anomaly_ratio', type=int, default=3, help='Max ratio of normal to anomaly patches')
    parser.add_argument('--min_tumor_ratio_in_patch', type=float, default=0.05, help='Min tumor ratio in anomalous patches')
    parser.add_argument('--anomaly_labels', type=int, nargs='+', default=[1, 2, 4], help='BraTS labels considered anomalies (default: 1 2 4)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    config = Config()
    config.num_subjects = args.num_subjects
    config.patch_size = args.patch_size
    config.patches_per_volume = args.patches_per_volume
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.embedding_dim = args.embedding_dim
    config.codebook_size = args.codebook_size
    config.commitment_beta = args.commitment_beta
    config.output_dir = create_unique_results_dir('vqvae_brats')
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir) if os.path.basename(_script_dir) == 'src' else _script_dir
    config.dataset_path = args.dataset_path if os.path.isabs(args.dataset_path) else os.path.join(_project_root, args.dataset_path)
    config.max_normal_to_anomaly_ratio = args.max_normal_to_anomaly_ratio
    config.min_tumor_ratio_in_patch = args.min_tumor_ratio_in_patch
    config.anomaly_labels = args.anomaly_labels
    config.verbose = args.verbose

    anomaly_names = _format_anomaly_labels(config.anomaly_labels, verbose=config.verbose)

    if config.verbose:
        print("=" * 60)
        print("3D VQ-VAE ANOMALY DETECTION FOR BRATS DATASET")
        print("=" * 60)
        print(f"Device: {config.device}")
        print(f"Dataset path: {config.dataset_path}")
        print(f"Output directory: {config.output_dir}")
        print(f"Patch size: {config.patch_size}^3")
        print(f"Patches per volume: {config.patches_per_volume}")
        print(f"VQ: embedding_dim={config.embedding_dim}, codebook_size={config.codebook_size}, beta={config.commitment_beta}")
        print(f"Anomaly labels: {anomaly_names}")
        print("=" * 60)
    else:
        print(f"3D VQ-VAE | Anomaly labels: {anomaly_names} | Output: {config.output_dir}")

    # Step 1: Process dataset and extract patches
    if config.verbose:
        print("\n1. Processing dataset and extracting patches...")
    processor = BraTSPreprocessor(config)
    patches, labels, subjects = processor.process_dataset(config.num_subjects)
    if len(patches) == 0:
        print("Error: No patches extracted")
        return
    if config.verbose:
        print(f"Total patches extracted: {len(patches)}")
        print(f"Patch shape: {patches[0].shape}")
        print(f"Normal patches: {np.sum(labels == 0)}")
        print(f"Anomalous patches: {np.sum(labels == 1)}")

    # Step 2: Subject-level split and unsupervised constraints
    if config.verbose:
        print("\n2. Subject-level data splitting")
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        print("ERROR: Dataset contains only one class! Cannot perform anomaly detection.")
        return
    unique_subjects = list(set(subjects))
    np.random.shuffle(unique_subjects)
    n_subjects = len(unique_subjects)
    train_subjects = unique_subjects[:int(0.6 * n_subjects)]
    val_subjects = unique_subjects[int(0.6 * n_subjects):int(0.8 * n_subjects)]
    test_subjects = unique_subjects[int(0.8 * n_subjects):]
    train_indices = [i for i, subj in enumerate(subjects) if subj in train_subjects]
    val_indices = [i for i, subj in enumerate(subjects) if subj in val_subjects]
    test_indices = [i for i, subj in enumerate(subjects) if subj in test_subjects]
    X_train_all = patches[train_indices]; y_train_all = labels[train_indices]
    X_val_all = patches[val_indices]; y_val_all = labels[val_indices]
    X_test = patches[test_indices]; y_test = labels[test_indices]
    train_normal_mask = (y_train_all == 0)
    val_normal_mask = (y_val_all == 0)
    X_train_normal = X_train_all[train_normal_mask]
    y_train_normal = y_train_all[train_normal_mask]
    X_val_normal = X_val_all[val_normal_mask]
    y_val_normal = y_val_all[val_normal_mask]

    train_dataset = BraTSPatchDataset(X_train_normal, y_train_normal)
    val_dataset = BraTSPatchDataset(X_val_normal, y_val_normal)
    test_dataset = BraTSPatchDataset(X_test, y_test)



    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    if config.verbose:
        print("\n4. Training 3D VQ-VAE...")
    detector = VQVAEAnomalyDetector(config)
    train_losses, val_losses = detector.train(train_loader, val_loader)

    if config.verbose:
        print("\n5. Evaluating model on test set...")
    results = detector.evaluate(test_loader, val_loader)

    if config.verbose:
        print("\n6. Creating visualizations...")
    visualizer = Visualizer(config)
    visualizer.create_summary_report(results)


    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    time_str = []
    if hours > 0:
        time_str.append(f"{hours}h")
    if minutes > 0:
        time_str.append(f"{minutes}m")
    time_str.append(f"{seconds}s")
    time_formatted = " ".join(time_str)
    results_file = os.path.join(config.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("3D VQ-VAE Anomaly Detection Results\n")
        f.write("="*60 + "\n")
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
        f.write(f"Configuration:\n")
        f.write(f"  Patch size: {config.patch_size}\n")
        f.write(f"  Patches per volume: {config.patches_per_volume}\n")
        f.write(f"  VQ embedding dim: {config.embedding_dim}\n")
        f.write(f"  Codebook size: {config.codebook_size}\n")
        f.write(f"  Commitment beta: {config.commitment_beta}\n")
        f.write(f"  Learning rate: {config.learning_rate}\n")
        f.write(f"  Batch size: {config.batch_size}\n")
        f.write(f"  Number of epochs: {config.num_epochs}\n")
        f.write(f"  Training samples: {len(train_dataset)}\n")
        f.write(f"  Validation samples: {len(val_dataset)}\n")
        f.write(f"  Test samples: {len(test_dataset)}\n")
        f.write(f"  Anomaly labels used: {config.anomaly_labels}\n")
        f.write("="*60 + "\n")
        f.write(f"EXECUTION TIME:\n")
        f.write(f"  Total time: {time_formatted} ({total_time:.1f} seconds)\n")
    print(f"\nPipeline completed in {time_formatted}. Results saved to: {results_file}")


if __name__ == "__main__":
    main()