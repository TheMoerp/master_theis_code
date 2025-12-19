#!/usr/bin/env python3

import os
import argparse
import random
import warnings
import time
import heapq
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.figure_factory as ff

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    confusion_matrix,
    roc_curve, precision_recall_curve, auc as sk_auc
)
from common.metrics import evaluate_binary_classification
from common.brats_preprocessing import BraTSPreprocessor, validate_patch_quality, create_unique_results_dir
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
    """Central configuration class for all parameters"""
    
    def __init__(self):
        # Dataset paths
        self.dataset_path = "datasets/BraTS2025-GLI-PRE-Challenge-TrainingData"
        self.output_dir = "ae_brats_results"
        
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
        
        # Segmentation labels for anomaly detection
        self.anomaly_labels = [1, 2, 4]
        
        # Brain tissue quality parameters for normal patches
        self.min_brain_tissue_ratio = 0.3
        self.max_background_intensity = 0.1
        self.min_brain_mean_intensity = 0.1
        self.max_high_intensity_ratio = 0.7
        self.high_intensity_threshold = 0.9
        self.edge_margin = 8
        
        # Model parameters
        self.latent_dim = 256
        self.learning_rate = 5e-5
        self.batch_size = 8
        self.num_epochs = 100
        self.early_stopping_patience = 20
        
        # Training parameters
        self.train_test_split = 0.8
        self.validation_split = 0.2
        
        # Visualization parameters
        self.slice_axis = 'axial'  # 'axial', 'coronal', 'sagittal'


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4 if torch.cuda.is_available() else 0



class BraTSPatchDataset(Dataset):
    """Dataset class for 3D patches from BraTS data"""
    
    def __init__(self, patches: np.ndarray, labels: np.ndarray, transform=None):
        self.patches = patches
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        
        # convedt to tensor and add channel dimension
        patch = torch.FloatTensor(patch).unsqueeze(0)
        label = torch.FloatTensor([label])
        
        if self.transform:
            patch = self.transform(patch)
            
        return patch, label


class Autoencoder3D(nn.Module):
    """Enhanced 3D Convolutional Autoencoder with skip connections for better anomaly detection"""
    
    def __init__(self, input_channels=1, latent_dim=128):
        super(Autoencoder3D, self).__init__()
        
        # Encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(2)
        
        self.encoder_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(2)
        
        self.encoder_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool3d(2) 
        
        self.encoder_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool3d(2)
        
        # Latent space
        self._latent_spatial_dim = 2  # 32 -> 16 -> 8 -> 4 -> 2 
        flattened_dim = 256 * (self._latent_spatial_dim ** 3)
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Sequential(
            nn.Linear(flattened_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, flattened_dim),
            nn.ReLU(inplace=True)
        )
        self.unflatten = nn.Unflatten(1, (256, self._latent_spatial_dim, self._latent_spatial_dim, self._latent_spatial_dim))


        # Decoder
        self.decoder_conv4 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
        self.decoder_conv3 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
        self.decoder_conv2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
        self.decoder_conv1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        # Encode
        e1 = self.encoder_conv1(x)
        e1_pool = self.pool1(e1)
        
        e2 = self.encoder_conv2(e1_pool)
        e2_pool = self.pool2(e2)
        
        e3 = self.encoder_conv3(e2_pool)
        e3_pool = self.pool3(e3)
        
        e4 = self.encoder_conv4(e3_pool)
        e4_pool = self.pool4(e4)
        
        # Representation
        flat = self.flatten(e4_pool)
        latent = self.fc_encode(flat)
        
        return latent
    
    def decode(self, latent):
        decoded = self.fc_decode(latent)
        unflat = self.unflatten(decoded)
        
        # Decode
        d4_prep = self.upsample4(unflat)
        d4 = self.decoder_conv4(d4_prep)
        d4_up = self.upsample3(d4)
        
        d3 = self.decoder_conv3(d4_up)
        d3_up = self.upsample2(d3)
        
        d2 = self.decoder_conv2(d3_up)
        d2_up = self.upsample1(d2)
        
        output = self.decoder_conv1(d2_up)
        
        return output
        
    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class AnomalyDetector:
    """Class for training and evaluating the autoencoder"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = Autoencoder3D(latent_dim=config.latent_dim).to(config.device)
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
        
        total_steps = self.config.num_epochs * len(train_loader)
        
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
                        reconstructed, _ = self.model(normal_data)
                        loss = criterion(reconstructed, normal_data)
                    
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
                
                avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
                train_losses.append(avg_train_loss)
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_normal_samples = 0
                
                with torch.no_grad():
                    for data, labels in val_loader:
                        data = data.to(self.config.device)
                        labels = labels.to(self.config.device)
                        
                        normal_mask = (labels == 0).squeeze()
                        
                        if normal_mask.sum() == 0:
                            continue
                            
                        normal_data = data[normal_mask]
                        val_normal_samples += normal_data.size(0)
                        
                        with autocast():
                            reconstructed, _ = self.model(normal_data)
                            loss = criterion(reconstructed, normal_data)
                        
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
                val_losses.append(avg_val_loss)
                
                scheduler.step(avg_val_loss)
                
                if (epoch + 1) % 10 == 0 and self.config.verbose:
                    print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
                    print(f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                    print(f"Normal samples processed: Train={normal_samples_processed}, Val={val_normal_samples}")
                
                # Early stop
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 
                              os.path.join(self.config.output_dir, 'best_autoencoder_3d.pth'))
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    pbar.write(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        if self.config.verbose:
            print(f"\nTraining completed.")
            print(f"Best validation loss: {best_val_loss:.6f}")
        
        # Save training history
        self.save_training_plots(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def save_training_plots(self, train_losses: List[float], val_losses: List[float]):
        """Save training and validation loss plots"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'training_losses.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def calculate_reconstruction_errors(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate reconstruction errors for anomaly detection"""
        self.model.eval()
        
        reconstruction_errors = []
        true_labels = []
        latent_features = []
        
        with torch.no_grad():
            for data, labels in tqdm(data_loader, desc="Calculating reconstruction errors"):
                data = data.to(self.config.device)
                
                with autocast():
                    reconstructed, latent = self.model(data)
                    
                mse = torch.mean((data - reconstructed) ** 2, dim=(1, 2, 3, 4))
                
                reconstruction_errors.extend(mse.cpu().numpy())
                true_labels.extend(labels.cpu().numpy().flatten())
                latent_features.extend(latent.cpu().numpy())
        
        return np.array(reconstruction_errors), np.array(true_labels), np.array(latent_features)
    
    def collect_reconstruction_examples(
        self,
        data_loader: DataLoader,
        num_good: int = 10,
        num_bad: int = 10
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Collect lowest/highest reconstruction-error samples separately for normal/anomaly labels.
        returns {'good': {'normal': [...], 'anomaly': [...]}, 'bad': {...}}
        """
        self.model.eval()
        label_map = {0: 'normal', 1: 'anomaly'}
        good_heaps = {lbl: [] for lbl in label_map.values()}
        bad_heaps = {lbl: [] for lbl in label_map.values()}
        example_counter = 0
        
        with torch.no_grad():
            for data, labels in tqdm(data_loader, desc="Selecting reconstruction examples"):
                data = data.to(self.config.device)
                labels = labels.to(self.config.device)
                
                with autocast():
                    recon, _ = self.model(data)
                
                diff = data - recon
                mse = torch.mean(diff ** 2, dim=(1, 2, 3, 4))
                residual = torch.abs(diff)
                
                for i in range(data.size(0)):
                    error_value = float(mse[i].item())
                    label_int = int(labels[i].item())
                    label_key = label_map.get(label_int, 'normal')
                    sample = {
                        'error': error_value,
                        'label': label_int,
                        'real': data[i:i+1].detach().cpu().numpy(),
                        'recon': recon[i:i+1].detach().cpu().numpy(),
                        'residual': residual[i:i+1].detach().cpu().numpy()
                    }

                    good_heap = good_heaps[label_key]
                    good_entry = (-error_value, example_counter, sample)
                    if len(good_heap) < num_good:
                        heapq.heappush(good_heap, good_entry)
                    else:
                        if -error_value > good_heap[0][0]:
                            heapq.heapreplace(good_heap, good_entry)
                    
                    bad_heap = bad_heaps[label_key]
                    bad_entry = (error_value, example_counter, sample)
                    if len(bad_heap) < num_bad:
                        heapq.heappush(bad_heap, bad_entry)
                    else:
                        if error_value > bad_heap[0][0]:
                            heapq.heapreplace(bad_heap, bad_entry)
                    
                    example_counter += 1
        
        def format_good(heap: List[Tuple[float, int, Dict]]):
            return [entry[2] for entry in sorted(heap, key=lambda x: x[0], reverse=True)]
        
        def format_bad(heap: List[Tuple[float, int, Dict]]):
            return [entry[2] for entry in sorted(heap, key=lambda x: x[0], reverse=True)]
        
        return {
            'good': {lbl: format_good(heap) for lbl, heap in good_heaps.items()},
            'bad': {lbl: format_bad(heap) for lbl, heap in bad_heaps.items()}
        }
    
    def find_unsupervised_threshold(self, val_loader: DataLoader) -> float:
        """Find threshold using ONLY normal validation data"""
        if self.config.verbose:
            print(f"\nTHRESHOLD DETERMINATION")
            print(f"{'='*60}")
            print("Computing threshold using ONLY normal validation data...")
        
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
                    reconstructed, _ = self.model(normal_data)
                    
                mse = torch.mean((normal_data - reconstructed) ** 2, dim=(1, 2, 3, 4))
                normal_val_errors.extend(mse.cpu().numpy())
        
        normal_val_errors = np.array(normal_val_errors)
        
        if self.config.verbose:
            print(f"Normal validation errors - Count: {len(normal_val_errors)}")
            print(f"Normal validation errors - Mean: {normal_val_errors.mean():.6f}, Std: {normal_val_errors.std():.6f}")
            print(f"Normal validation errors - Min: {normal_val_errors.min():.6f}, Max: {normal_val_errors.max():.6f}")
        
        threshold_methods = {
            'percentile_95': np.percentile(normal_val_errors, 95),
            'percentile_97': np.percentile(normal_val_errors, 97),
            'percentile_99': np.percentile(normal_val_errors, 99),
            'mean_plus_2std': normal_val_errors.mean() + 2 * normal_val_errors.std(),
            'mean_plus_3std': normal_val_errors.mean() + 3 * normal_val_errors.std(),
            'median_plus_2mad': np.median(normal_val_errors) + 2 * np.median(np.abs(normal_val_errors - np.median(normal_val_errors))),
            'iqr_outlier': np.percentile(normal_val_errors, 75) + 1.5 * (np.percentile(normal_val_errors, 75) - np.percentile(normal_val_errors, 25))
        }
        
        if self.config.verbose:
            print(f"THRESHOLD CANDIDATES:")
            print(f"{'Method':<20} {'Threshold':<12}")
            print(f"{'-'*35}")
            for method, threshold in threshold_methods.items():
                print(f"{method:<20} {threshold:<12.6f}")
        
        selected_threshold = threshold_methods['percentile_95']
        selected_method = 'percentile_95'
        
        if self.config.verbose:
            print(f"\nSELECTED THRESHOLD: {selected_threshold:.6f} (Method: {selected_method})")
            print(f"Rationale: 95th percentile of normal validation errors is a conservative,")
            print(f"commonly used threshold that doesn't require test label access.")
            print(f"{'='*60}")
        
        return selected_threshold
    
    def evaluate(self, test_loader: DataLoader, val_loader: DataLoader) -> Dict:
        # Load best model
        model_path = os.path.join(self.config.output_dir, 'best_autoencoder_3d.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            if self.config.verbose:
                print("Loaded best model for evaluation")
        
        optimal_threshold = self.find_unsupervised_threshold(val_loader)
        
        reconstruction_errors, true_labels, latent_features = self.calculate_reconstruction_errors(test_loader)
        reconstruction_examples = self.collect_reconstruction_examples(test_loader, num_good=10, num_bad=10)
        
        print(f"\nEVALUATION: CAE Results")
        print(f"{'='*60}")
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
        
        print(f"\nCAE PERFORMANCE:")
        print(f"{'='*60}")
        print(f"ROC AUC:                  {roc_auc:.4f}")
        print(f"Average Precision (AP):   {average_precision:.4f}")
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
        print(f"{'='*60}")
        
        print(f"\nPOST-HOC RECONSTRUCTION ERROR ANALYSIS:")
        print(f"Normal errors    - Mean: {normal_errors.mean():.6f}, Std: {normal_errors.std():.6f}")
        print(f"Anomaly errors   - Mean: {anomaly_errors.mean():.6f}, Std: {anomaly_errors.std():.6f}")
        print(f"Separation ratio - Anomaly/Normal: {separation_ratio:.3f}")
            
        print(f"\nCONFUSION MATRIX ANALYSIS:")
        print(f"True Negatives:      {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives:                  {fn}")
        print(f"True Positives:       {tp}")
        
        results = {
            'reconstruction_errors': reconstruction_errors,
            'true_labels': true_labels,
            'predictions': predictions,
            'latent_features': latent_features,
            'reconstruction_examples': reconstruction_examples,
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
    """Class for creating various visualizations"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def plot_confusion_matrix(self, true_labels: np.ndarray, predictions: np.ndarray):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        
        if self.config.verbose:
            print(f"\nConfusion Matrix Details:")
            print(f"True Labels - Normal: {np.sum(true_labels == 0)}, Anomaly: {np.sum(true_labels == 1)}")
            print(f"Predictions - Normal: {np.sum(predictions == 0)}, Anomaly: {np.sum(predictions == 1)}")
            print(f"Confusion Matrix:\n{cm}")

        tn, fp, fn, tp = cm.ravel()
        
        z = [[tn, fp], [fn, tp]]
        x = ['Normal (0)', 'Anomaly (1)']
        y = ['Normal (0)', 'Anomaly (1)']

        row_sums = cm.sum(axis=1)
        
        # Avoid division by zero
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

        total_samples = cm.sum()
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
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
            xref='paper',
            yref='paper',
            x=0.0,
            y=-0.28,
            bordercolor="black",
            borderwidth=1,
            bgcolor="lightgray",
            font_size=12
        )
        
        fig.update_layout(margin=dict(t=100, b=150))

        output_path = os.path.join(self.config.output_dir, 'confusion_matrix.png')
        try:
            fig.write_image(output_path, width=800, height=800, scale=2)
            if self.config.verbose:
                print(f"Confusion Matrix plot saved to {output_path}")
        except ValueError as e: 
            print(f"ERROR: Could not save confusion matrix plot: {e}")
    
    def plot_roc_curve(self, true_labels: np.ndarray, reconstruction_errors: np.ndarray):
        """Plot ROC curve"""
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
        plt.savefig(os.path.join(self.config.output_dir, 'roc_curve.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, true_labels: np.ndarray, reconstruction_errors: np.ndarray):
        """Plot Precision-Recall curve"""
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
        plt.savefig(os.path.join(self.config.output_dir, 'precision_recall_curve.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_reconstruction_error_histogram(self, reconstruction_errors: np.ndarray, 
                                          true_labels: np.ndarray, optimal_threshold: float):
        """Plot histogram of reconstruction errors"""
        normal_errors = reconstruction_errors[true_labels == 0]
        anomaly_errors = reconstruction_errors[true_labels == 1]
        
        plt.figure(figsize=(12, 6))
        
        plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        plt.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, 
                   label=f'Optimal Threshold = {optimal_threshold:.6f}')
        
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'reconstruction_error_histogram.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_latent_space_visualization(self, latent_features: np.ndarray, true_labels: np.ndarray):
        """Plot t-SNE and PCA visualizations of latent space"""
        print("Creating latent space visualizations...")
        
        # Limit samples for visualization if too many
        if len(latent_features) > 2000:
            indices = np.random.choice(len(latent_features), 2000, replace=False)
            latent_features = latent_features[indices]
            true_labels = true_labels[indices]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # PCA
        print("Computing PCA...")
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(latent_features)
        
        scatter1 = ax1.scatter(pca_features[true_labels == 0, 0], pca_features[true_labels == 0, 1], 
                             c='blue', alpha=0.6, label='Normal', s=20)
        scatter2 = ax1.scatter(pca_features[true_labels == 1, 0], pca_features[true_labels == 1, 1], 
                             c='red', alpha=0.6, label='Anomaly', s=20)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title('PCA of Latent Features')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # t-SNE
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_features)//4))
        tsne_features = tsne.fit_transform(latent_features)
        
        scatter3 = ax2.scatter(tsne_features[true_labels == 0, 0], tsne_features[true_labels == 0, 1], 
                             c='blue', alpha=0.6, label='Normal', s=20)
        scatter4 = ax2.scatter(tsne_features[true_labels == 1, 0], tsne_features[true_labels == 1, 1], 
                             c='red', alpha=0.6, label='Anomaly', s=20)
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.set_title('t-SNE of Latent Features')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'latent_space_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_slice_axis_index(self) -> int:
        axis = getattr(self.config, 'slice_axis', 'axial')
        axis = axis.lower() if isinstance(axis, str) else 'axial'
        axis_map = {
            'sagittal': 4,  # width axis
            'coronal': 3,   # height axis
            'axial': 2      # depth axis
        }
        return axis_map.get(axis, 2)
    
    def _extract_slice(self, volume: np.ndarray, axis_idx: int, slice_idx: int) -> np.ndarray:
        if axis_idx == 2:
            return volume[0, 0, slice_idx, :, :]
        if axis_idx == 3:
            return volume[0, 0, :, slice_idx, :]
        if axis_idx == 4:
            return volume[0, 0, :, :, slice_idx]
        # Fallback to axial
        return volume[0, 0, slice_idx, :, :]
    
    def _plot_reconstruction_pairs(
        self,
        axes,
        real: np.ndarray,
        recon: np.ndarray,
        axis_idx: int,
        slice_fracs: List[float]
    ):
        axis_size = real.shape[axis_idx]
        for col, frac in enumerate(slice_fracs):
            slice_idx = max(0, min(axis_size - 1, int(axis_size * frac)))
            real_slice = self._extract_slice(real, axis_idx, slice_idx)
            recon_slice = self._extract_slice(recon, axis_idx, slice_idx)
            
            axes[0, col].imshow(real_slice, cmap='gray')
            axes[0, col].set_title('Input')
            axes[0, col].axis('off')
            
            axes[1, col].imshow(recon_slice, cmap='gray')
            axes[1, col].set_title('Recon')
            axes[1, col].axis('off')
    
    def save_reconstruction_examples(self, example_sets: Dict[str, Dict[str, List[Dict]]]):
        """Save slices for good/bad reconstruction quality"""
        if not example_sets:
            return
        
        axis_idx = self._get_slice_axis_index()
        slice_fracs = [0.35, 0.5, 0.65]
        base_dir = os.path.join(self.config.output_dir, 'reconstruction_examples')
        os.makedirs(base_dir, exist_ok=True)
        
        for category in ['good', 'bad']:
            label_dict = example_sets.get(category, {})
            if not label_dict:
                continue
            
            for label_key in ['normal', 'anomaly']:
                samples = label_dict.get(label_key, [])
                if not samples:
                    continue
                
                category_dir = os.path.join(base_dir, f'{category}_examples', label_key)
                os.makedirs(category_dir, exist_ok=True)
                
                for idx, sample in enumerate(samples, 1):
                    fig, axes = plt.subplots(2, len(slice_fracs), figsize=(12, 7))
                    self._plot_reconstruction_pairs(
                        axes,
                        sample['real'],
                        sample['recon'],
                        axis_idx,
                        slice_fracs
                    )
                    label_text = "Normal" if label_key == 'normal' else "Anomaly"
                    fig.suptitle(
                        f"{category.capitalize()} reconstruction #{idx} | {label_text} | MSE: {sample['error']:.6f}",
                        fontsize=14
                    )
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    output_path = os.path.join(category_dir, f"{category}_{label_key}_{idx:03d}.png")
                    plt.savefig(output_path, dpi=200, bbox_inches='tight')
                    plt.close()
    
    def visualize_3d_patches(self, patches: np.ndarray, labels: np.ndarray, 
                           num_normal: int = 100, num_anomaly: int = 10):

        patches_dir = os.path.join(self.config.output_dir, 'patches_visualization')
        os.makedirs(patches_dir, exist_ok=True)

        normal_dir = os.path.join(patches_dir, 'normal_patches')
        anomaly_dir = os.path.join(patches_dir, 'anomaly_patches')
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(anomaly_dir, exist_ok=True)
        
        if self.config.verbose:
            print(f"Visualizing patches: {patches_dir}")
            print(f"Saving {num_normal} normal patches and {num_anomaly} anomaly patches...")
        
        normal_indices = np.where(labels == 0)[0]
        anomaly_indices = np.where(labels == 1)[0]
        
        if len(normal_indices) > 0:
            selected_normal = np.random.choice(normal_indices, 
                                             size=min(num_normal, len(normal_indices)), 
                                             replace=False)
            
            for i, idx in enumerate(tqdm(selected_normal, desc="Saving normal patches")):
                patch = patches[idx]
                
                # Create subplot for different slice orientations
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                fig.suptitle(f'Normal Patch #{i+1}', fontsize=16, fontweight='bold')
                
                # Show axial slices
                for j in range(4):
                    slice_idx = int(patch.shape[2] * (j + 1) / 5) 
                    axes[0, j].imshow(patch[:, :, slice_idx], cmap='gray')
                    axes[0, j].set_title(f'Axial Slice {slice_idx}')
                    axes[0, j].axis('off')
                
                # Show coronal slices
                for j in range(4):
                    slice_idx = int(patch.shape[1] * (j + 1) / 5) 
                    axes[1, j].imshow(patch[:, slice_idx, :], cmap='gray')
                    axes[1, j].set_title(f'Coronal Slice {slice_idx}')
                    axes[1, j].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(normal_dir, f'normal_patch_{i+1:03d}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        # Sample anomaly patches
        if len(anomaly_indices) > 0:
            selected_anomaly = np.random.choice(anomaly_indices, 
                                              size=min(num_anomaly, len(anomaly_indices)), 
                                              replace=False)
            
            for i, idx in enumerate(tqdm(selected_anomaly, desc="Saving anomaly patches")):
                patch = patches[idx]
                
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                fig.suptitle(f'Anomaly Patch #{i+1}', fontsize=16, fontweight='bold', color='red')
                
                for j in range(4):
                    slice_idx = int(patch.shape[2] * (j + 1) / 5) 
                    axes[0, j].imshow(patch[:, :, slice_idx], cmap='gray')
                    axes[0, j].set_title(f'Axial Slice {slice_idx}')
                    axes[0, j].axis('off')
            
                for j in range(4):
                    slice_idx = int(patch.shape[1] * (j + 1) / 5) 
                    axes[1, j].imshow(patch[:, slice_idx, :], cmap='gray')
                    axes[1, j].set_title(f'Coronal Slice {slice_idx}')
                    axes[1, j].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(anomaly_dir, f'anomaly_patch_{i+1:03d}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        if self.config.verbose:
            print(f"Normal patches saved to: {normal_dir}")
            print(f"Anomaly patches saved to: {anomaly_dir}")
            print(f"Total patches visualized: {min(num_normal, len(normal_indices))} normal + {min(num_anomaly, len(anomaly_indices))} anomaly")
    
    def create_summary_report(self, results: Dict):
        self.plot_confusion_matrix(results['true_labels'], results['predictions'])
        self.plot_roc_curve(results['true_labels'], results['reconstruction_errors'])
        self.plot_precision_recall_curve(results['true_labels'], results['reconstruction_errors'])
        self.plot_reconstruction_error_histogram(results['reconstruction_errors'], 
                                                results['true_labels'], 
                                                results['optimal_threshold'])
        self.plot_latent_space_visualization(results['latent_features'], results['true_labels'])
        if results.get('reconstruction_examples'):
            self.save_reconstruction_examples(results['reconstruction_examples'])
        
        print(f"\nAll visualizations saved to: {self.config.output_dir}")


def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='CAE Anomaly Detection for BraTS Dataset')
    parser.add_argument('--num_subjects', type=int, default=None, 
                       help='Number of subjects to use (default: all)')
    parser.add_argument('--patch_size', type=int, default=32, 
                       help='Size of 3D patches (default: 32)')
    parser.add_argument('--patches_per_volume', type=int, default=50, 
                       help='Number of patches per volume (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size for training (default: 8)')
    parser.add_argument('--num_epochs', type=int, default=100, 
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, 
                       help='Learning rate (default: 5e-5)')
    parser.add_argument('--latent_dim', type=int, default=256, 
                       help='Latent dimension size (default: 256)')
    parser.add_argument('--output_dir', type=str, default='ae_brats_results', 
                       help='Output directory (default: ae_brats_results)')
    parser.add_argument('--dataset_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData', 
                       help='Path to BraTS dataset')
    parser.add_argument('--max_normal_to_anomaly_ratio', type=int, default=3,
                       help='Maximum ratio of normal to anomaly patches (default: 3)')
    parser.add_argument('--min_tumor_ratio_in_patch', type=float, default=0.05,
                       help='Minimum tumor ratio in anomalous patches (default: 0.05)')
    parser.add_argument('--anomaly_labels', type=int, nargs='+', default=[1, 2, 4],
                       help='BraTS segmentation labels to consider as anomalies (default: [1, 2, 4]). '
                            'Available labels: 0=background/normal tissue, 1=NCR/NET (necrotic and non-enhancing tumor core), '
                            '2=ED (peritumoral edematous/invaded tissue), 4=ET (GD-enhancing tumor). '
                            'Examples: --anomaly_labels 1 4 (only solid tumor), --anomaly_labels 2 (only edema), '
                            '--anomaly_labels 1 2 4 (all tumor types - default)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output with detailed debug information (default: minimal output)')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    config.num_subjects = args.num_subjects
    config.patch_size = args.patch_size
    config.patches_per_volume = args.patches_per_volume
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.latent_dim = args.latent_dim
    config.output_dir = create_unique_results_dir('ae_brats')
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir) if os.path.basename(_script_dir) == 'src' else _script_dir
    config.dataset_path = args.dataset_path if os.path.isabs(args.dataset_path) else os.path.join(_project_root, args.dataset_path)
    config.max_normal_to_anomaly_ratio = args.max_normal_to_anomaly_ratio
    config.min_tumor_ratio_in_patch = args.min_tumor_ratio_in_patch
    config.anomaly_labels = args.anomaly_labels
    config.verbose = args.verbose
    
    # Explicitly inform the user whether CUDA is being used
    if config.device.type == 'cuda':
        device_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_idx)
        print(f"CUDA enabled (device {device_idx}): {device_name}")
    else:
        print("CUDA not available - falling back to CPU.")
    
    
    label_names = {
        0: "Background/Normal",
        1: "NCR/NET (Necrotic/Non-enhancing)",
        2: "ED (Edema)",
        4: "ET (Enhancing Tumor)"
    }
    anomaly_names = [
        f"{label}: {label_names.get(label, 'Unknown')}" for label in config.anomaly_labels
    ]

    if config.verbose:
        print("="*60)
        print("3DCAE ANOMALY DETECTION FOR BRATS DATASET")
        print("="*60)
        print(f"Device: {config.device}")
        print(f"Dataset path: {config.dataset_path}")
        print(f"Output directory: {config.output_dir}")
        print(f"Patch size: {config.patch_size}x{config.patch_size}x{config.patch_size}")
        print(f"Patches per volume: {config.patches_per_volume}")
        print(f"Number of subjects: {config.num_subjects if config.num_subjects else 'All'}")
        
        print(f"3D Autoencoder Anomaly Detection | Anomaly labels: {anomaly_names} | Output: {config.output_dir}")
    
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
        
        validate_patch_quality(patches, labels, verbose=config.verbose)
    
    # Step 2: Split data into train and test sets
    if config.verbose:
        print("\n2. Data splitting")
    
    # Check if we have both classes
    unique_labels, counts = np.unique(labels, return_counts=True)
    if config.verbose:
        print(f"Label distribution: {dict(zip(unique_labels, counts))}")
    
    if len(unique_labels) < 2:
        print("ERROR: Dataset contains only one class")
        return
    
    subject_label_map: Dict[str, set] = {}
    for subj, label in zip(subjects, labels):
        subject_label_map.setdefault(subj, set()).add(int(label))
    unique_subjects = list(subject_label_map.keys())
    if config.verbose:
        print(f"Total unique subjects: {len(unique_subjects)}")
    
    anomaly_subjects = [subj for subj, lbls in subject_label_map.items() if 1 in lbls]
    normal_only_subjects = [subj for subj, lbls in subject_label_map.items() if 1 not in lbls]
    rng = np.random.default_rng(42)
    rng.shuffle(anomaly_subjects)
    rng.shuffle(normal_only_subjects)
    
    n_subjects = len(unique_subjects)
    train_target = max(1, int(0.6 * n_subjects))
    val_target = max(1, int(0.2 * n_subjects))
    test_target = max(1, n_subjects - train_target - val_target)
    if train_target + val_target + test_target > n_subjects:
        train_target = n_subjects - val_target - test_target
        train_target = max(train_target, 1)
    
    def allocate_subjects(target_count: int, require_anomaly: bool = False):
        assigned = []
        needs_anomaly = require_anomaly and target_count > 0
        if needs_anomaly and anomaly_subjects:
            assigned.append(anomaly_subjects.pop())
            needs_anomaly = False
        while len(assigned) < target_count and (anomaly_subjects or normal_only_subjects):
            if normal_only_subjects and (len(normal_only_subjects) >= len(anomaly_subjects)):
                assigned.append(normal_only_subjects.pop())
            elif anomaly_subjects:
                assigned.append(anomaly_subjects.pop())
            else:
                break
        return assigned, needs_anomaly
    
    val_subjects, val_missing_anom = allocate_subjects(val_target, require_anomaly=True)
    test_subjects, test_missing_anom = allocate_subjects(test_target, require_anomaly=True)
    
    if val_missing_anom and config.verbose:
        print("WARNING: Not enough anomaly subjects to include in validation split.")
    if test_missing_anom and config.verbose:
        print("WARNING: Not enough anomaly subjects to include in test split.")
    
    remaining_subjects = anomaly_subjects + normal_only_subjects
    rng.shuffle(remaining_subjects)
    train_subjects = remaining_subjects
    
    if not train_subjects:
        # Steal from validation or test if possible
        donor = val_subjects if len(val_subjects) > len(test_subjects) else test_subjects
        if donor:
            train_subjects.append(donor.pop())
    
    if config.verbose:
        print(f"Subject distribution:")
        print(f"  Training subjects: {len(train_subjects)}")
        print(f"  Validation subjects: {len(val_subjects)}")
        print(f"  Test subjects: {len(test_subjects)}")
    
    train_indices = [i for i, subj in enumerate(subjects) if subj in train_subjects]
    val_indices = [i for i, subj in enumerate(subjects) if subj in val_subjects]
    test_indices = [i for i, subj in enumerate(subjects) if subj in test_subjects]
    
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
    
    
    train_dataset = BraTSPatchDataset(X_train_normal, y_train_normal)
    val_dataset = BraTSPatchDataset(X_val_normal, y_val_normal)
    test_dataset = BraTSPatchDataset(X_test, y_test)
    
    # Step 3: Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=config.num_workers)
    
    # Step 4: Initialize and train the autoencoder
    if config.verbose:
        print("\n4. Training 3D Autoencoder...")
    detector = AnomalyDetector(config)
    train_losses, val_losses = detector.train(train_loader, val_loader)
    
    # Step 5: Evaluate the model
    if config.verbose:
        print("\n5. Evaluating model on test set...")
    results = detector.evaluate(test_loader, val_loader)
    
    # Step 6: Create visualizations
    if config.verbose:
        print("\n6. Creating visualizations...")
    visualizer = Visualizer(config)
    visualizer.create_summary_report(results)
    
    # Step 7: Visualize sample patches
    if config.verbose:
        print("\n7. Visualizing sample patches...")
    visualizer.visualize_3d_patches(X_test, y_test, num_normal=100, num_anomaly=10)
    
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
    
    # Save results to file
    results_file = os.path.join(config.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("3DCAE Anomaly Detection Results\n")
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
        f.write(f"  Latent dimension: {config.latent_dim}\n")
        f.write(f"  Learning rate: {config.learning_rate}\n")
        f.write(f"  Batch size: {config.batch_size}\n")
        f.write(f"  Number of epochs: {config.num_epochs}\n")
        f.write(f"  Training samples: {len(X_train_normal)}\n")
        f.write(f"  Validation samples: {len(X_val_normal)}\n")
        f.write(f"  Test samples: {len(X_test)}\n")
        f.write(f"  Training subjects: {len(train_subjects)}\n")
        f.write(f"  Validation subjects: {len(val_subjects)}\n")
        f.write(f"  Test subjects: {len(test_subjects)}\n")
        f.write(f"  Anomaly labels used: {config.anomaly_labels}\n")
        f.write(f"  Anomaly label descriptions: {anomaly_names}\n")
        f.write("="*60 + "\n")
        f.write(f"EXECUTION TIME:\n")
        f.write(f"  Total time: {time_formatted} ({total_time:.1f} seconds)\n")
    
    if config.verbose:
        print(f"\nResults saved to: {results_file}")
        print("\n" + "="*60)
        print(f"Total execution time: {time_formatted} ({total_time:.1f} seconds)")
        print("="*60)
    else:
        print(f"\nPipeline completed in {time_formatted}. Results saved to: {results_file}")


if __name__ == "__main__":
    main() 

