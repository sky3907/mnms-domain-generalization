import ast
import hashlib
import json
import os
import shutil
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from medpy.metric.binary import hd95
from monai.networks.nets import UNet
from monai.transforms import Resize
from scipy.stats import skew, kurtosis

from dataset import build_dataset
from train_augmented_matched_v1 import MatchedUNet

ROOT = Path(__file__).resolve().parent.parent
DRIVE = Path('/content/drive/MyDrive')
DATASET_ROOT = '/content/OpenDataset'
REFERENCE = ROOT / 'src/evaluate_prompt_comparison.py'
CLUSTER_DIR = ROOT / 'latent_domain_discovery'
RUN_DIR = DRIVE / 'mnms_augmented_training_v1_seed43'
LOCAL_OUTPUT = ROOT / 'augmented_evaluation_v1_seed43'
DRIVE_OUTPUT = DRIVE / 'mnms_augmented_evaluation_v1_seed43'
MODELS = ('source_only', 'generic_prompt', 'latent_domain_prompt')
METRICS = ('Dice_LV', 'HD95_LV', 'Dice_Myocardium', 'HD95_Myocardium',
           'Dice_RV', 'HD95_RV', 'Mean_Dice', 'Mean_HD95')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_verified(df, filename):
    local = LOCAL_OUTPUT / filename
    remote = DRIVE_OUTPUT / filename
    if local.exists() or remote.exists():
        raise FileExistsError(f'Output already exists: {filename}')
    df.to_csv(local, index=False)
    with remote.with_name(remote.name + '.copying').open('wb') as dst:
        with local.open('rb') as src:
            shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    temporary = remote.with_name(remote.name + '.copying')
    if sha256(local) != sha256(temporary):
        raise RuntimeError(f'Backup checksum mismatch: {filename}')
    temporary.replace(remote)
    if sha256(local) != sha256(remote):
        raise RuntimeError(f'Final checksum mismatch: {filename}')
    print('Saved and verified:', remote)


def load_exact_evaluator_functions(namespace):
    tree = ast.parse(REFERENCE.read_text(encoding='utf-8'))
    names = (
        'resize_pred_to_native', 'normalize_slice', 'get_robust_voxels',
        'compute_entropy', 'extract_style_features', 'assign_latent_cluster',
        'prepare_volume_tensor', 'run_source_inference',
        'run_prompt_inference', 'compute_volume_metrics',
    )
    nodes = []
    for name in names:
        found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(found) != 1:
            raise RuntimeError(f'Expected exactly one definition of {name}')
        nodes.append(found[0])
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(REFERENCE), 'exec'), namespace)


class SourceAdapter(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x):
        unused = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.core(x, unused)


def main():
    if not DRIVE.is_dir():
        raise RuntimeError('Mount Google Drive first')
    files = [REFERENCE, ROOT / 'src/train_augmented_matched_v1.py',
             CLUSTER_DIR / 'clustering_metadata.json',
             CLUSTER_DIR / 'style_scaler.joblib',
             CLUSTER_DIR / 'style_pca.joblib',
             CLUSTER_DIR / 'style_kmeans.joblib']
    for name in MODELS:
        files.extend([RUN_DIR / name / 'best_model.pth',
                      RUN_DIR / name / 'training_summary.csv'])
    for f in files:
        if not f.is_file() or f.stat().st_size == 0:
            raise FileNotFoundError(f'Missing or empty: {f}')
    for folder in (LOCAL_OUTPUT, DRIVE_OUTPUT):
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f'Existing evaluation outputs: {folder}; do not overwrite')
    LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    DRIVE_OUTPUT.mkdir(parents=True, exist_ok=True)
    print('Device:', DEVICE)
    print('Reusing unchanged evaluation functions from:', REFERENCE)

    cluster_metadata = json.loads((CLUSTER_DIR / 'clustering_metadata.json').read_text())
    if (cluster_metadata['selected_k'] != 2 or
            cluster_metadata['vendor_centre_used_for_clustering'] is not False):
        raise RuntimeError('Unexpected clustering pipeline')
    scaler = joblib.load(CLUSTER_DIR / 'style_scaler.joblib')
    pca = joblib.load(CLUSTER_DIR / 'style_pca.joblib')
    kmeans = joblib.load(CLUSTER_DIR / 'style_kmeans.joblib')

    ns = dict(
        __builtins__=__builtins__, np=np, torch=torch, nn=nn,
        UNet=UNet, Resize=Resize, hd95=hd95, skew=skew, kurtosis=kurtosis,
        DEVICE=DEVICE, INFERENCE_CHUNK_SIZE=32,
        resize_image_to_model=Resize(spatial_size=(256, 256)),
        feature_columns=cluster_metadata['feature_columns'],
        n_pca_components=int(cluster_metadata['pca_components_95']),
        scaler=scaler, pca=pca, kmeans=kmeans,
        STRUCTURE_NAMES=['LV', 'Myocardium', 'RV'],
    )
    load_exact_evaluator_functions(ns)

    models = {}
    for name in MODELS:
        record = pd.read_csv(RUN_DIR / name / 'training_summary.csv')
        if (len(record) != 1 or record.iloc[0]['model'] != name or
                str(record.iloc[0]['complete']).lower() != 'true' or
                int(record.iloc[0]['seed']) != 43):
            raise RuntimeError(f'Incomplete or inconsistent training summary: {name}')
        model = MatchedUNet(name).to(DEVICE)
        weights = torch.load(RUN_DIR / name / 'best_model.pth',
                             map_location=DEVICE, weights_only=True)
        model.load_state_dict(weights, strict=True)
        model.eval()
        models[name] = model
        print(f'Loaded {name}: best A/B Dice {record.iloc[0]["best_dice"]:.6f}')

    ns['source_model'] = SourceAdapter(models['source_only']).eval()
    val_samples = build_dataset(split='val', root_dir=DATASET_ROOT)
    counts = pd.Series([s['vendor'] for s in val_samples]).value_counts().to_dict()
    if counts != {'A': 8, 'B': 20, 'C': 20, 'D': 20}:
        raise RuntimeError(f'Unexpected official validation distribution: {counts}')

    rows = []
    for i, sample in enumerate(val_samples, start=1):
        image_obj = nib.load(sample['image_path'])
        mask_obj = nib.load(sample['mask_path'])
        image = image_obj.get_fdata(dtype=np.float32)
        masks = mask_obj.get_fdata(dtype=np.float32)
        frame = int(sample['frame_idx'])
        phase_volume = image[:, :, :, frame]
        gt = masks[:, :, :, frame].astype(np.int64)
        spacing = tuple(float(x) for x in image_obj.header.get_zooms()[:3])
        cluster = ns['assign_latent_cluster'](phase_volume)
        batch, native_hw, n_slices = ns['prepare_volume_tensor'](image, frame)

        predictions = {
            'source_only': ns['run_source_inference'](batch, native_hw, n_slices),
            'generic_prompt': ns['run_prompt_inference'](
                models['generic_prompt'], batch, native_hw, n_slices, 0),
            'latent_domain_prompt': ns['run_prompt_inference'](
                models['latent_domain_prompt'], batch, native_hw, n_slices, cluster),
        }
        for name, pred in predictions.items():
            if pred.shape != gt.shape:
                raise RuntimeError(f'Shape mismatch: {name}, {sample["patient"]}')
            metrics = ns['compute_volume_metrics'](pred, gt, spacing)
            rows.append({
                'model': name, 'patient': str(sample['patient']),
                'phase': str(sample['phase']), 'vendor': sample['vendor'],
                'domain': 'in-domain' if sample['vendor'] in ('A', 'B') else 'unseen',
                'latent_cluster': cluster, **metrics,
            })
        print(f'{i:02d}/68  {sample["patient"]} {sample["phase"]} '
              f'Vendor {sample["vendor"]} completed', flush=True)

    result = pd.DataFrame(rows)
    if (len(result) != 204 or
            result.duplicated(['model', 'patient', 'phase']).any()):
        raise RuntimeError('Unexpected or duplicate patient-phase results')

    domain = (result.groupby(['model', 'domain'], sort=False)[list(METRICS)]
              .mean().reset_index())
    vendor = (result.groupby(['model', 'vendor'], sort=False)[list(METRICS)]
              .mean().reset_index())
    validity = (result.groupby(['model', 'vendor'], sort=False)[list(METRICS)]
                .count().reset_index())

    unseen = result[result.vendor.isin(['C', 'D'])]
    wide = unseen.pivot(index=['patient', 'phase', 'vendor'], columns='model',
                        values=['Mean_Dice', 'Mean_HD95'])
    wide.columns = [f'{metric}_{name}' for metric, name in wide.columns]
    wide = wide.reset_index()
    if len(wide) != 40 or wide.isna().any().any():
        raise RuntimeError('Missing matched unseen-vendor measurements')
    wide['latent_Dice_gain_vs_source'] = (
        wide['Mean_Dice_latent_domain_prompt'] - wide['Mean_Dice_source_only'])
    wide['latent_HD95_reduction_vs_source'] = (
        wide['Mean_HD95_source_only'] - wide['Mean_HD95_latent_domain_prompt'])

    save_verified(result, 'patient_level_results.csv')
    save_verified(domain, 'model_domain_summary.csv')
    save_verified(vendor, 'model_vendor_summary.csv')
    save_verified(validity, 'metric_valid_counts.csv')
    save_verified(wide, 'paired_unseen_comparison.csv')

    metadata = {
        'description': 'Frozen augmented seed43 checkpoints; official validation evaluation',
        'checkpoint_selection': 'Internal A/B only; no C/D tuning or selection',
        'inference_source': str(REFERENCE),
        'inference_source_sha256': sha256(REFERENCE),
        'training_script_sha256': sha256(ROOT / 'src/train_augmented_matched_v1.py'),
        'checkpoint_sha256': {name: sha256(RUN_DIR / name / 'best_model.pth')
                              for name in MODELS},
        'metric_policy': 'Original evaluator compute_volume_metrics; NaNs excluded in means',
        'samples_per_vendor': counts,
    }
    local_meta = LOCAL_OUTPUT / 'evaluation_metadata.json'
    remote_meta = DRIVE_OUTPUT / 'evaluation_metadata.json'
    local_meta.write_text(json.dumps(metadata, indent=2))
    shutil.copy2(local_meta, remote_meta)
    if sha256(local_meta) != sha256(remote_meta):
        raise RuntimeError('Metadata backup failed')

    print('\n=== AUGMENTED NATIVE-VOLUME RESULTS ===')
    print(domain[['model', 'domain', 'Mean_Dice', 'Mean_HD95']].to_string(index=False))
    print('\n=== PAIRED C/D RESULTS ===')
    print('Mean Dice gain, latent vs source:',
          wide['latent_Dice_gain_vs_source'].mean())
    print('Mean HD95 reduction, latent vs source:',
          wide['latent_HD95_reduction_vs_source'].mean())
    print('Files:', DRIVE_OUTPUT)
    print('No model training or target-based checkpoint selection performed.')


if __name__ == '__main__':
    main()
