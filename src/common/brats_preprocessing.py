import os
import glob
from typing import List, Tuple, Optional, Dict

import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import uniform_filter, zoom, gaussian_filter


PERTURBATION_LEVELS = {
    'weak': {
        'gaussian_std': 0.02,
        'motion_blur_size': 3,
        'low_resolution_scale': 0.85,
        'bias_field_amplitude': 0.12,
    },
    'medium': {
        'gaussian_std': 0.05,
        'motion_blur_size': 5,
        'low_resolution_scale': 0.65,
        'bias_field_amplitude': 0.22,
    },
    'strong': {
        'gaussian_std': 0.08,
        'motion_blur_size': 7,
        'low_resolution_scale': 0.45,
        'bias_field_amplitude': 0.35,
    },
}


def _clip_unit(volume: np.ndarray) -> np.ndarray:
    return np.clip(volume, 0.0, 1.0).astype(np.float32)


def _add_gaussian_noise(volume: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    noisy = volume + rng.normal(0.0, std, size=volume.shape)
    return _clip_unit(noisy)


def _apply_motion_blur(volume: np.ndarray, kernel_size: int) -> np.ndarray:
    blurred = uniform_filter(volume, size=kernel_size, mode='nearest')
    return _clip_unit(blurred)


def _simulate_low_resolution(volume: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return volume.astype(np.float32)
    scale_factors = (scale, scale, scale)
    low_res = zoom(volume, zoom=scale_factors, order=1)
    # Restore to original resolution
    zoom_back = tuple(orig / float(res) for orig, res in zip(volume.shape, low_res.shape))
    restored = zoom(low_res, zoom=zoom_back, order=1)
    # Ensure shape consistency
    if restored.shape != volume.shape:
        pad_width = [(0, max(0, o - r)) for r, o in zip(restored.shape, volume.shape)]
        restored = np.pad(restored, pad_width, mode='edge')
        slices = tuple(slice(0, o) for o in volume.shape)
        restored = restored[slices]
    return _clip_unit(restored)


def _apply_bias_field(volume: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    coarse_shape = tuple(max(2, s // 8) for s in volume.shape)
    field = rng.normal(0.0, 1.0, size=coarse_shape)
    field = gaussian_filter(field, sigma=1.0)
    zoom_back = tuple(orig / float(res) for orig, res in zip(volume.shape, field.shape))
    field = zoom(field, zoom=zoom_back, order=3, mode='nearest')
    if field.shape != volume.shape:
        pad_width = [(0, max(0, o - r)) for r, o in zip(field.shape, volume.shape)]
        field = np.pad(field, pad_width, mode='edge')
        slices = tuple(slice(0, o) for o in volume.shape)
        field = field[slices]
    field = field - field.min()
    denom = field.max() - field.min() + 1e-8
    field = field / denom
    field = 1.0 + amplitude * (field - 0.5) * 2.0
    biased = volume * field
    return _clip_unit(biased)


def _project_root_from_script_dir(script_dir: str) -> str:
    return os.path.dirname(script_dir) if os.path.basename(script_dir) == 'src' else script_dir


def resolve_dataset_path(dataset_path: str, script_dir: str) -> str:
    """Resolve dataset path relative to project root when given as relative path."""
    if os.path.isabs(dataset_path):
        return dataset_path
    project_root = _project_root_from_script_dir(script_dir)
    return os.path.join(project_root, dataset_path)


def create_unique_results_dir(model_tag: str, script_dir: str = None) -> str:
    """Create results/results_<model_tag>[N] under project root (dedup with numeric suffix).

    If script_dir is None, infer project root from this module location (src/common/..).
    """
    if script_dir is None:
        # src/common -> src -> project_root
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(module_dir))
    else:
        project_root = _project_root_from_script_dir(script_dir)
    base_results = os.path.join(project_root, 'results')
    os.makedirs(base_results, exist_ok=True)
    base_name = f"results_{model_tag}"
    candidate = os.path.join(base_results, base_name)
    if not os.path.exists(candidate):
        os.makedirs(candidate)
        return candidate
    suffix = 2
    while True:
        cand = os.path.join(base_results, f"{base_name}{suffix}")
        if not os.path.exists(cand):
            os.makedirs(cand)
            return cand
        suffix += 1


def subject_level_split(subjects: List[str], train_frac: float = 0.6, val_frac: float = 0.2):
    """Split subject ids into train/val/test sets by subject"""
    import random

    uniq = list(set(subjects))
    random.shuffle(uniq)
    n = len(uniq)
    train_subj = set(uniq[:int(train_frac * n)])
    val_subj = set(uniq[int(train_frac * n):int((train_frac + val_frac) * n)])
    test_subj = set(uniq[int((train_frac + val_frac) * n):])
    idx_train = [i for i, s in enumerate(subjects) if s in train_subj]
    idx_val = [i for i, s in enumerate(subjects) if s in val_subj]
    idx_test = [i for i, s in enumerate(subjects) if s in test_subj]
    return idx_train, idx_val, idx_test, train_subj, val_subj, test_subj


class BraTSPreprocessor:
    """Preprocessing and patch extraction for BraTS volumes.
    """

    def __init__(self, config):
        self.config = config

    # Defaults helper
    def _get(self, name: str, default):
        return getattr(self.config, name, default)

    def load_volume(self, subject_path: str, modality: str = 't1c') -> Tuple[np.ndarray, np.ndarray]:
        volume_file = glob.glob(os.path.join(subject_path, f"*-{modality}.nii.gz"))[0]
        seg_file = glob.glob(os.path.join(subject_path, "*-seg.nii.gz"))
        if len(seg_file) == 0:
            seg_file = glob.glob(os.path.join(subject_path, "*seg.nii.gz"))
        seg_file = seg_file[0]
        volume = nib.load(volume_file).get_fdata()
        segmentation = nib.load(seg_file).get_fdata()
        return volume, segmentation

    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        volume = volume.astype(np.float32)
        non_zero_mask = volume > 0
        if np.sum(non_zero_mask) == 0:
            return volume
        v = volume[non_zero_mask]
        p1 = np.percentile(v, 1)
        p99 = np.percentile(v, 99)
        volume = np.clip(volume, p1, p99)
        volume[non_zero_mask] = (volume[non_zero_mask] - p1) / (p99 - p1 + 1e-8)
        return np.clip(volume, 0.0, 1.0)

    def _is_brain_tissue_patch(self, patch: np.ndarray) -> bool:
        max_background_intensity = self._get('max_background_intensity', 0.1)
        min_brain_tissue_ratio = self._get('min_brain_tissue_ratio', 0.3)
        min_brain_mean_intensity = self._get('min_brain_mean_intensity', 0.1)
        high_intensity_threshold = self._get('high_intensity_threshold', 0.9)
        max_high_intensity_ratio = self._get('max_high_intensity_ratio', 0.7)
        min_patch_std = self._get('min_patch_std', 0.01)

        brain_mask = patch > max_background_intensity
        if brain_mask.mean() < min_brain_tissue_ratio:
            return False
        vals = patch[brain_mask]
        if vals.size == 0:
            return False
        if vals.mean() < min_brain_mean_intensity:
            return False
        high_mask = patch > high_intensity_threshold
        if high_mask.mean() > max_high_intensity_ratio:
            return False
        if patch.std() < min_patch_std * 2:
            return False
        reasonable = ((patch > 0.05) & (patch < 0.95)).mean()
        return reasonable >= 0.5

    def _anomaly_ratio(self, seg_patch: np.ndarray) -> float:
        labels = np.array(self._get('anomaly_labels', [1, 2, 4]))
        return np.isin(seg_patch, labels).sum() / seg_patch.size

    def extract_normal_patches(self, volume: np.ndarray, seg: np.ndarray) -> List[np.ndarray]:
        patches: List[np.ndarray] = []
        max_background_intensity = self._get('max_background_intensity', 0.1)
        max_tumor_ratio_normal = self._get('max_tumor_ratio_normal', 0.01)
        min_patch_std = self._get('min_patch_std', 0.01)
        edge = self._get('edge_margin', 8)
        ps = self._get('patch_size', 32)
        max_normal_patches_per_subject = self._get('max_normal_patches_per_subject', 100)

        brain_mask = (volume > max_background_intensity) & (volume < 0.95)
        anomaly_mask = np.isin(seg, self._get('anomaly_labels', [1, 2, 4]))
        coords_normal = np.where(~anomaly_mask)
        coords_brain = np.where(brain_mask)
        valid = list(set(zip(*coords_normal)).intersection(set(zip(*coords_brain))))
        if not valid:
            return patches
        sx, sy, sz = volume.shape
        filtered = []
        for x, y, z in valid:
            if x < edge or y < edge or z < edge or x >= sx - edge or y >= sy - edge or z >= sz - edge:
                continue
            x0, y0, z0 = x - ps // 2, y - ps // 2, z - ps // 2
            x1, y1, z1 = x0 + ps, y0 + ps, z0 + ps
            if x0 >= 0 and y0 >= 0 and z0 >= 0 and x1 <= sx and y1 <= sy and z1 <= sz:
                filtered.append((x, y, z))
        if not filtered:
            return patches
        max_patches = min(len(filtered) // 20, max_normal_patches_per_subject)
        max_patches = max(max_patches, 10)
        sample_n = min(max_patches * 5, len(filtered))
        idxs = np.random.choice(len(filtered), size=sample_n, replace=False)
        for i in tqdm(idxs, desc="Extracting normal patches", leave=False):
            x, y, z = filtered[i]
            x0, y0, z0 = x - ps // 2, y - ps // 2, z - ps // 2
            x1, y1, z1 = x0 + ps, y0 + ps, z0 + ps
            patch = volume[x0:x1, y0:y1, z0:z1]
            seg_patch = seg[x0:x1, y0:y1, z0:z1]
            if self._anomaly_ratio(seg_patch) > max_tumor_ratio_normal:
                continue
            if not self._is_brain_tissue_patch(patch):
                continue
            if patch.std() < min_patch_std:
                continue
            patches.append(patch)
            if len(patches) >= max_patches:
                break
        return patches

    def extract_anomalous_patches(self, volume: np.ndarray, seg: np.ndarray) -> List[np.ndarray]:
        patches: List[np.ndarray] = []
        ps = self._get('patch_size', 32)
        max_anomaly_patches_per_subject = self._get('max_anomaly_patches_per_subject', 50)
        min_patch_std = self._get('min_patch_std', 0.01)
        min_patch_mean = self._get('min_patch_mean', 0.05)
        min_tumor_ratio_anomaly = self._get('min_tumor_ratio_anomaly', 0.05)
        anomaly_coords = np.where(np.isin(seg, self._get('anomaly_labels', [1, 2, 4])))
        if len(anomaly_coords[0]) == 0:
            return patches
        sx, sy, sz = volume.shape
        max_patches = min(len(anomaly_coords[0]) // 50, max_anomaly_patches_per_subject)
        if max_patches == 0:
            return patches
        idxs = np.random.choice(len(anomaly_coords[0]), size=min(max_patches, len(anomaly_coords[0])), replace=False)
        for i in tqdm(idxs, desc="Extracting anomaly patches", leave=False):
            x, y, z = anomaly_coords[0][i], anomaly_coords[1][i], anomaly_coords[2][i]
            x0, y0, z0 = max(0, x - ps // 2), max(0, y - ps // 2), max(0, z - ps // 2)
            x1, y1, z1 = min(sx, x0 + ps), min(sy, y0 + ps), min(sz, z0 + ps)
            if (x1 - x0 != ps) or (y1 - y0 != ps) or (z1 - z0 != ps):
                continue
            patch = volume[x0:x1, y0:y1, z0:z1]
            if patch.std() <= min_patch_std or patch.mean() <= min_patch_mean:
                continue
            seg_patch = seg[x0:x1, y0:y1, z0:z1]
            if self._anomaly_ratio(seg_patch) >= min_tumor_ratio_anomaly:
                patches.append(patch)
        return patches

    def process_dataset(self, num_subjects: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        subject_dirs = [d for d in os.listdir(self.config.dataset_path) if os.path.isdir(os.path.join(self.config.dataset_path, d))]
        if num_subjects:
            subject_dirs = subject_dirs[:num_subjects]
        all_normal, all_anom, subj_normal, subj_anom = [], [], [], []
        for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
            sp = os.path.join(self.config.dataset_path, subject_dir)
            try:
                vol, seg = self.load_volume(sp)
                vol = self.normalize_volume(vol)
                normals = self.extract_normal_patches(vol, seg)
                anoms = self.extract_anomalous_patches(vol, seg)
                all_normal.extend(normals)
                all_anom.extend(anoms)
                subj_normal.extend([subject_dir] * len(normals))
                subj_anom.extend([subject_dir] * len(anoms))
            except Exception:
                continue
        max_ratio = self._get('max_normal_to_anomaly_ratio', 3)
        max_normal = int(len(all_anom) * max_ratio) if len(all_anom) > 0 else len(all_normal)
        if len(all_normal) > max_normal and max_normal > 0:
            idx = np.random.choice(len(all_normal), max_normal, replace=False)
            all_normal = [all_normal[i] for i in idx]
            subj_normal = [subj_normal[i] for i in idx]
        patches = all_normal + all_anom
        labels = [0] * len(all_normal) + [1] * len(all_anom)
        subjects = subj_normal + subj_anom
        return np.array(patches, dtype=np.float32), np.array(labels, dtype=np.int64), subjects

    def generate_test_perturbations(
        self,
        patches: np.ndarray,
        severity: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        if patches.size == 0:
            return {}

        level = severity or self._get('test_perturbation', None)
        if level is None:
            return {}
        level = str(level).lower()
        if level in ('none', 'off', 'disable', 'disabled'):
            return {}
        if level not in PERTURBATION_LEVELS:
            raise ValueError(f"Unsupported perturbation severity '{level}'. "
                             f"Choose from {list(PERTURBATION_LEVELS.keys())}.")

        params = PERTURBATION_LEVELS[level]
        seed = random_seed if random_seed is not None else self._get('test_perturbation_seed', None)
        rng = np.random.default_rng(seed)

        perturbations = {}

        # Individual perturbations
        gaussian_set = [_add_gaussian_noise(patch, params['gaussian_std'], rng) for patch in patches]
        perturbations['gaussian_noise'] = np.stack(gaussian_set, axis=0)

        blur_set = [_apply_motion_blur(patch, params['motion_blur_size']) for patch in patches]
        perturbations['motion_blur'] = np.stack(blur_set, axis=0)

        low_res_set = [_simulate_low_resolution(patch, params['low_resolution_scale']) for patch in patches]
        perturbations['low_resolution'] = np.stack(low_res_set, axis=0)

        bias_set = [_apply_bias_field(patch, params['bias_field_amplitude'], rng) for patch in patches]
        perturbations['bias_field'] = np.stack(bias_set, axis=0)

        combined = []
        for patch in patches:
            perturbed = _add_gaussian_noise(patch, params['gaussian_std'], rng)
            perturbed = _apply_motion_blur(perturbed, params['motion_blur_size'])
            perturbed = _simulate_low_resolution(perturbed, params['low_resolution_scale'])
            perturbed = _apply_bias_field(perturbed, params['bias_field_amplitude'], rng)
            combined.append(perturbed)
        perturbations['combined'] = np.stack(combined, axis=0)

        return {k: v.astype(np.float32) for k, v in perturbations.items()}

def validate_patch_quality(patches: np.ndarray, labels: np.ndarray, verbose: bool = False):
    normal = patches[labels == 0]
    anomaly = patches[labels == 1]
    stats = {
        'normal_patches': {
            'count': int(len(normal)),
            'mean_intensity': float(np.mean([p.mean() for p in normal])) if len(normal) else 0.0,
            'std_intensity': float(np.std([p.mean() for p in normal])) if len(normal) else 0.0,
            'non_zero_ratio': float(np.mean([(p > 0).sum() / p.size for p in normal])) if len(normal) else 0.0,
        },
        'anomaly_patches': {
            'count': int(len(anomaly)),
            'mean_intensity': float(np.mean([p.mean() for p in anomaly])) if len(anomaly) else 0.0,
            'std_intensity': float(np.std([p.mean() for p in anomaly])) if len(anomaly) else 0.0,
            'non_zero_ratio': float(np.mean([(p > 0).sum() / p.size for p in anomaly])) if len(anomaly) else 0.0,
        },
    }
    if verbose:
        print("Normal patches statistics:")
        print(f"  Count: {stats['normal_patches']['count']}")
        print(f"  Mean intensity: {stats['normal_patches']['mean_intensity']:.4f}")
        print(f"  Non-zero ratio: {stats['normal_patches']['non_zero_ratio']:.4f}")
        print("Anomaly patches statistics:")
        print(f"  Count: {stats['anomaly_patches']['count']}")
        print(f"  Mean intensity: {stats['anomaly_patches']['mean_intensity']:.4f}")
        print(f"  Non-zero ratio: {stats['anomaly_patches']['non_zero_ratio']:.4f}")
    return stats
