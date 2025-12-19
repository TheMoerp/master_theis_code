#!/usr/bin/env python3

import os
import sys
import argparse
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
import common.brats_preprocessing as br_prep
from common.brats_preprocessing import BraTSPreprocessor, create_unique_results_dir
from common.metrics import evaluate_binary_classification, threshold_from_percentile
from sklearn.decomposition import PCA
import plotly.figure_factory as ff
from sklearn.manifold import TSNE


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
        self.output_dir = "anatomix_knn_results"

        # Patch extraction parameters
        self.patch_size = 32
        self.patches_per_volume = 50
        self.min_non_zero_ratio = 0.2
        self.max_normal_to_anomaly_ratio = 3
        self.min_tumor_ratio_in_patch = 0.05

        # Patch quality parameters
        self.min_patch_std = 0.01
        self.min_patch_mean = 0.05
        self.max_tumor_ratio_normal = 0.01
        self.min_tumor_ratio_anomaly = 0.05
        self.max_normal_patches_per_subject = 100
        self.max_anomaly_patches_per_subject = 50

        self.anomaly_labels = [1, 4]

        # Brain tissue quality parameters
        self.min_brain_tissue_ratio = 0.3
        self.max_background_intensity = 0.1
        self.min_brain_mean_intensity = 0.1
        self.max_high_intensity_ratio = 0.7
        self.high_intensity_threshold = 0.9
        self.edge_margin = 8

        # Splits
        self.train_test_split = 0.8
        self.validation_split = 0.2

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4 if torch.cuda.is_available() else 0

        # Anatomix + feature parameters
        self.anatomix_batch_size = 4
        self.feature_pooling = "mean_max_std"

        # KNN params
        self.k_neighbors = 7

        # Threshold percentile
        self.threshold_percentile = 90.0

        os.makedirs(self.output_dir, exist_ok=True)


def install_anatomix_if_needed():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'src' else script_dir
    repo_path = os.path.join(project_root, 'anatomix')
    pkg_root = os.path.join(repo_path, 'anatomix')
    if os.path.isdir(pkg_root):
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            from anatomix.model.network import Unet
            return True
        except Exception:
            pass
    # Fallback: try editable install if local not importable
    try:
        from anatomix.model.network import Unet 
        return True
    except Exception:
        print("Could not import anatomix. Ensure the local repo exists at ./anatomix or install it.")
        return False


class AnatomixFeatureExtractor:
    def __init__(self, device: torch.device, pooling: str = "mean_max_std"):
        self.device = device
        self.pooling = pooling
        from anatomix.model.network import Unet

        # init anatomxi 
        self.model = Unet(
            dimension=3,
            input_nc=1,
            output_nc=16,
            num_downs=4,
            ngf=16,
        )

        # Load weights
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'src' else script_dir
        repo_path = os.path.join(project_root, 'anatomix')
        weights_primary = os.path.join(repo_path, 'model-weights', 'anatomix.pth')
        weights_brains = os.path.join(repo_path, 'model-weights', 'anatomix+brains.pth')
        weights_path = weights_primary if os.path.isfile(weights_primary) else (weights_brains if os.path.isfile(weights_brains) else None)
        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state, strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device)

    @staticmethod
    def _standardize_size(patch: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        out = np.zeros(target_size, dtype=patch.dtype)
        sx, sy, sz = patch.shape
        tx, ty, tz = target_size
        xs = max(0, (sx - tx) // 2); xe = xs + min(tx, sx)
        ys = max(0, (sy - ty) // 2); ye = ys + min(ty, sy)
        zs = max(0, (sz - tz) // 2); ze = zs + min(tz, sz)
        xsp = max(0, (tx - sx) // 2); ysp = max(0, (ty - sy) // 2); zsp = max(0, (tz - sz) // 2)
        out[xsp:xsp + (xe - xs), ysp:ysp + (ye - ys), zsp:zsp + (ze - zs)] = patch[xs:xe, ys:ye, zs:ze]
        return out

    def extract_batch(self, patches: List[np.ndarray], patch_size: int) -> np.ndarray:

        standardized = [self._standardize_size(p, (patch_size, patch_size, patch_size)) for p in patches]
        x = torch.from_numpy(np.stack(standardized)[..., None].transpose(0, 4, 1, 2, 3)).float().to(self.device)
        with torch.no_grad():
            # Use encoder-only 
            feats = self.model(x, encode_only=True)
        f = feats.detach().cpu().numpy()
        f = np.transpose(f, (0, 2, 3, 4, 1))
        
        # Global pooling
        mean = f.mean(axis=(1, 2, 3))
        mx = f.max(axis=(1, 2, 3))
        sd = f.std(axis=(1, 2, 3))
        return np.concatenate([mean, mx, sd], axis=1)

    def extract_all(self, patches: np.ndarray, batch_size: int, patch_size: int) -> np.ndarray:
        features = []
        for i in tqdm(range(0, len(patches), batch_size), desc="Extracting Anatomix features"):
            batch = patches[i:i + batch_size]
            feats = self.extract_batch(list(batch), patch_size)
            features.append(feats)
        return np.vstack(features) if len(features) else np.zeros((0,))



def fit_knn_on_normal(train_features: np.ndarray, k: int):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k, len(train_features)), algorithm='auto')
    nbrs.fit(train_features)
    return nbrs


def knn_anomaly_scores(nbrs, ref_features: np.ndarray, query_features: np.ndarray, k: int) -> np.ndarray:
    distances, _ = nbrs.kneighbors(query_features, n_neighbors=min(k, len(ref_features)))
    return distances.mean(axis=1)


def threshold_from_normal(val_scores: np.ndarray, percentile: float) -> float:
    return threshold_from_percentile(val_scores, percentile)


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict:
    return evaluate_binary_classification(y_true, scores, threshold)


def save_results(config: Config, results: Dict, y_true: np.ndarray, scores: np.ndarray, threshold: float,
                 train_n: int, val_n: int, test_n: int,
                 train_subjects_n: int, val_subjects_n: int, test_subjects_n: int):
    out = os.path.join(config.output_dir, 'evaluation_results.txt')
    with open(out, 'w') as f:
        f.write("Anatomix+KNN Anomaly Detection Results\n")
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
        f.write(f"  Training samples: {train_n}\n")
        f.write(f"  Validation samples: {val_n}\n")
        f.write(f"  Test samples: {test_n}\n")
        f.write(f"  Training subjects: {train_subjects_n}\n")
        f.write(f"  Validation subjects: {val_subjects_n}\n")
        f.write(f"  Test subjects: {test_subjects_n}\n")
        f.write(f"  Anomaly labels used: {config.anomaly_labels}\n")
    print(f"Results saved to: {out}")


class Visualizer:
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
        roc_auc_value = roc_auc_score(true_labels, scores)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc_value:.2f})")
        plt.fill_between(fpr, tpr, step="pre", alpha=0.25, color="#aec7e8")
        plt.plot([0, 1], [0, 1], linestyle="--", color="#888888", lw=1, label="Chance")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        plt.savefig(os.path.join(self.config.output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_precision_recall_curve(self, true_labels: np.ndarray, scores: np.ndarray):
        precision, recall, _ = precision_recall_curve(true_labels, scores)
        pr_auc_value = auc(recall, precision)
        plt.figure(figsize=(7, 6))
        plt.plot(recall, precision, color="#ff7f0e", lw=2, label=f"PR (AUC = {pr_auc_value:.2f})")
        plt.fill_between(recall, precision, step="pre", alpha=0.25, color="#ffbb78")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
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

        plt.xlim(0, 2.5)
        plt.ylim(0, 3.5)

        plt.xticks(np.arange(0.0, 2.5 + 0.001, 0.5))
        plt.yticks(np.arange(0.0, 3.5 + 0.001, 0.5))
        plt.savefig(os.path.join(self.config.output_dir, 'score_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_latent_space_visualization(self, features: np.ndarray, true_labels: np.ndarray):
        print("Creating latent space visualizations...")

        feats = features
        labels = true_labels

        if len(feats) == 0 or feats.ndim != 2:
            return

        # damit tsne besser lesbar ist
        if len(feats) > 2000:
            idx = np.random.choice(len(feats), 2000, replace=False)
            feats = feats[idx]
            labels = labels[idx]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # PCA
        print("Computing PCA...")
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(feats)

        ax1.scatter(pca_features[labels == 0, 0], pca_features[labels == 0, 1],
                    c='blue', alpha=0.6, label='Normal', s=20)
        ax1.scatter(pca_features[labels == 1, 0], pca_features[labels == 1, 1],
                    c='red', alpha=0.6, label='Anomaly', s=20)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title('PCA of Latent Features')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # t-SNE
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(feats)//4)))
        tsne_features = tsne.fit_transform(feats)

        ax2.scatter(tsne_features[labels == 0, 0], tsne_features[labels == 0, 1],
                    c='blue', alpha=0.6, label='Normal', s=20)
        ax2.scatter(tsne_features[labels == 1, 0], tsne_features[labels == 1, 1],
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


def main():

    parser = argparse.ArgumentParser(description='Anatomix+KNN Unsupervised Anomaly Detection for BraTS')
    parser.add_argument('--num_subjects', type=int, default=None, help='Number of subjects to use (default: all)')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of 3D patches (default: 32)')
    parser.add_argument('--patches_per_volume', type=int, default=50, help='Number of patches per volume (default: 50)')
    parser.add_argument('--output_dir', type=str, default='anatomix_knn_results', help='Output directory')
    parser.add_argument('--dataset_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData', help='Path to BraTS dataset')
    parser.add_argument('--max_normal_to_anomaly_ratio', type=int, default=3, help='Max ratio of normal to anomaly patches')
    parser.add_argument('--min_tumor_ratio_in_patch', type=float, default=0.05, help='Minimum tumor ratio in anomalous patches')
    parser.add_argument('--anomaly_labels', type=int, nargs='+', default=[1, 4], help='BraTS labels to consider anomalous (default: 1 4)')
    parser.add_argument('--k_neighbors', type=int, default=7, help='Number of neighbors for KNN')
    parser.add_argument('--threshold_percentile', type=float, default=90.0, help='Percentile for threshold on validation NORMAL scores (0-100)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    cfg = Config()
    cfg.num_subjects = args.num_subjects
    cfg.patch_size = args.patch_size
    cfg.patches_per_volume = args.patches_per_volume
    cfg.output_dir = create_unique_results_dir('anatomix_knn_brats')
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir) if os.path.basename(_script_dir) == 'src' else _script_dir
    cfg.dataset_path = args.dataset_path if os.path.isabs(args.dataset_path) else os.path.join(_project_root, args.dataset_path)
    cfg.max_normal_to_anomaly_ratio = args.max_normal_to_anomaly_ratio
    cfg.min_tumor_ratio_in_patch = args.min_tumor_ratio_in_patch
    cfg.anomaly_labels = args.anomaly_labels
    cfg.k_neighbors = args.k_neighbors
    cfg.threshold_percentile = args.threshold_percentile
    cfg.verbose = args.verbose

    print(f"Anatomix+KNN | Anomaly labels: {cfg.anomaly_labels} | Output: {cfg.output_dir}")

    processor = BraTSPreprocessor(cfg)
    patches, labels, subjects = processor.process_dataset(cfg.num_subjects)
    if len(patches) == 0:
        print("No patches extracted. Exiting.")
        return

    idx_train, idx_val, idx_test, train_subj, val_subj, test_subj = br_prep.subject_level_split(subjects)
    X_train_all, y_train_all = patches[idx_train], labels[idx_train]
    X_val_all, y_val_all = patches[idx_val], labels[idx_val]
    X_test, y_test = patches[idx_test], labels[idx_test]

    train_normal_mask = (y_train_all == 0)
    val_normal_mask = (y_val_all == 0)
    X_train_normal = X_train_all[train_normal_mask]
    X_val_normal = X_val_all[val_normal_mask]

    print(f"Train normal: {len(X_train_normal)} | Val normal: {len(X_val_normal)} | Test total: {len(X_test)}")

    if not install_anatomix_if_needed():
        return
    extractor = AnatomixFeatureExtractor(cfg.device, pooling=cfg.feature_pooling)
    train_feats = extractor.extract_all(X_train_normal, cfg.anatomix_batch_size, cfg.patch_size)
    val_feats = extractor.extract_all(X_val_normal, cfg.anatomix_batch_size, cfg.patch_size)
    test_feats = extractor.extract_all(X_test, cfg.anatomix_batch_size, cfg.patch_size)

    knn = fit_knn_on_normal(train_feats, cfg.k_neighbors)

    val_scores = knn_anomaly_scores(knn, train_feats, val_feats, cfg.k_neighbors)
    test_scores = knn_anomaly_scores(knn, train_feats, test_feats, cfg.k_neighbors)

    thr = threshold_from_normal(val_scores, cfg.threshold_percentile)

    results = evaluate_scores(y_test, test_scores, thr)

    save_results(
        cfg, results, y_test, test_scores, thr,
        train_n=len(X_train_normal), val_n=len(X_val_normal), test_n=len(X_test),
        train_subjects_n=len(train_subj), val_subjects_n=len(val_subj), test_subjects_n=len(test_subj)
    )

    vis = Visualizer(cfg)
    vis.plot_confusion_matrix(y_test, results['predictions'])
    vis.plot_roc_curve(y_test, test_scores)
    vis.plot_precision_recall_curve(y_test, test_scores)
    vis.plot_score_histogram(test_scores, y_test, thr)
    vis.plot_latent_space_visualization(test_feats, y_test)


if __name__ == "__main__":
    main()