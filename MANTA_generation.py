#!/usr/bin/env python3
"""MANTA sequence-to-Cα ensemble generator.

Usage:
    python manta_generate.py "AMINO_ACID_SEQUENCE" output.pdb
    python MANTA_generation_rg_percent_v4.py \
        "AMINO_ACID_SEQUENCE" output.pdb \
        --target-rg 35 \
        --num-frames 1000 \
        --rg-std-scale 150
        
Required repository layout:
    manta_generate.py
    weght/MANTA.pth
"""

import os
import argparse
import gc
import hashlib
import random
import re
import time
import warnings
from pathlib import Path
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmModel
from Bio import BiopythonWarning
from Bio.PDB import Atom, Chain, Model, PDBIO, Residue, Structure
warnings.simplefilter('ignore', BiopythonWarning)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENCODER_CHECKPOINT = SCRIPT_DIR / 'weight' / 'MANTA.pth'
DEFAULT_ESM_MODEL = 'facebook/esm2_t33_650M_UR50D'
DEFAULT_NUM_FRAMES = 300
DEFAULT_DECODER_BATCH_SIZE = 300
DEFAULT_MAX_ITER = 600
DEFAULT_SEED = 42
DEFAULT_DEVICE = 'cuda'
RG_BASE_CALIB_SCALE = 1.0
RG_OPTUNA_MEAN_SCALE = 1.1101784249260092
RG_MEAN_SCALE = RG_BASE_CALIB_SCALE * RG_OPTUNA_MEAN_SCALE
RG_STD_BASE_SCALE = 0.30
DEFAULT_RG_STD_PERCENT = 100.0
APPLY_RG_CALIBRATION = True
DECODER_LAM = 275.7386526647752
DECODER_P = 6.91537362718835
DECODER_HUB_WEIGHT_SCALE = 0.32859914738338414
DECODER_SAMPLE_MIX = 0.9958379083638593

class EncoderConfig:
    ESM_MODEL_NAME = DEFAULT_ESM_MODEL
    MODEL_PATH = str(DEFAULT_ENCODER_CHECKPOINT)
    MAX_LEN = 1022
    D_EMB = 1280
    D_ATT = 4 * 20
    D_HIDDEN = 128
    OUT_BINS = 32
    BIN_MIN = 2.0
    BIN_MAX = 22.0
    AFFINITY_SIGMA = 12.0
    BOND_LENGTH = 3.8

def seed_everything(seed: int=DEFAULT_SEED) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def as_float_scalar(x, default=None):
    if x is None:
        return default
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        value = float(arr.reshape(-1)[0])
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default

def sequence_to_resnames(sequence: str) -> np.ndarray:
    aa1to3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
    return np.array([aa1to3.get(residue.upper(), 'GLY') for residue in sequence])

def resolve_device() -> torch.device:
    if DEFAULT_DEVICE == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required by the default configuration, but torch.cuda.is_available() returned False.')
    return torch.device(DEFAULT_DEVICE)

class TriangleMultiplicativeUpdate(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj_left = nn.Linear(d_model, d_model)
        self.proj_right = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.norm(z)
        mask_4d = mask.unsqueeze(-1)
        left = self.proj_left(z) * mask_4d
        right = self.proj_right(z) * mask_4d
        update = torch.einsum('bikd,bjkd->bijd', left, right)
        return z + self.proj_out(update)

class IDPFusionMultiTaskModel(nn.Module):

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.proj_1d = nn.Linear(cfg.D_EMB, cfg.D_HIDDEN)
        self.proj_2d = nn.Linear(cfg.D_ATT, cfg.D_HIDDEN)
        self.fusion = nn.Linear(cfg.D_HIDDEN * 3, cfg.D_HIDDEN)
        self.triangle_blocks = nn.ModuleList([TriangleMultiplicativeUpdate(cfg.D_HIDDEN) for _ in range(3)])
        self.prob_head = nn.Linear(cfg.D_HIDDEN, cfg.OUT_BINS)
        self.dist_head = nn.Linear(cfg.D_HIDDEN, 1)
        self.sep2_head = nn.Linear(cfg.D_HIDDEN, 1)
        self.rg_head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(cfg.D_HIDDEN, 1))

    def forward(self, emb: torch.Tensor, att: torch.Tensor, mask: torch.Tensor):
        batch_size, length, _ = emb.shape
        emb_proj = self.proj_1d(emb)
        emb_i = emb_proj.unsqueeze(2).expand(batch_size, length, length, -1)
        emb_j = emb_proj.unsqueeze(1).expand(batch_size, length, length, -1)
        att_proj = self.proj_2d(att)
        z = torch.cat([emb_i, emb_j, att_proj], dim=-1)
        z = self.fusion(z)
        for block in self.triangle_blocks:
            z = block(z, mask)
        z = 0.5 * (z + z.transpose(1, 2))
        log_probs = F.log_softmax(self.prob_head(z), dim=-1)
        pred_dist = F.softplus(self.dist_head(z)).squeeze(-1)
        if length >= 3:
            diag_idx = torch.arange(length - 2, device=z.device)
            pred_sep2 = F.softplus(self.sep2_head(z[:, diag_idx, diag_idx + 2])).squeeze(-1)
        else:
            pred_sep2 = z.new_empty((batch_size, 0))
        z_for_rg = z.permute(0, 3, 1, 2)
        pred_rg = F.softplus(self.rg_head(z_for_rg)).squeeze(-1)
        return (log_probs, pred_dist, pred_sep2, pred_rg, z)

class DirectESM2Extractor:

    def __init__(self, model_name: str, device: torch.device):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name, output_attentions=True)
        self.model.eval()
        self.model.to(self.device)

    @torch.inference_mode()
    def extract(self, sequence: str):
        inputs = self.tokenizer(sequence, return_tensors='pt', add_special_tokens=True).to(self.device)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[0, 1:-1, :].float()
        last_4_layers = outputs.attentions[-4:]
        stacked_attentions = torch.stack(last_4_layers).squeeze(1)
        attentions_4_20 = stacked_attentions[:, :, 1:-1, 1:-1].float()
        length = embeddings.shape[0]
        attentions = attentions_4_20.reshape(-1, length, length).permute(1, 2, 0).contiguous()
        return (embeddings, attentions)

class DirectEncoderPredictor:

    def __init__(self, cfg: EncoderConfig, checkpoint_path: Path, device: torch.device):
        self.cfg = cfg
        self.device = torch.device(device)
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Encoder checkpoint was not found:\n{checkpoint_path}\n\nlace MANTA.pth in the repository's weights/ directory.")
        self.model = IDPFusionMultiTaskModel(cfg)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if len(state_dict) > 0 and next(iter(state_dict.keys())).startswith('module.'):
            state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)

    @torch.inference_mode()
    def predict(self, emb: torch.Tensor, att: torch.Tensor) -> dict:
        length = int(emb.shape[0])
        emb = emb.unsqueeze(0).to(self.device, dtype=torch.float32)
        att = att.unsqueeze(0).to(self.device, dtype=torch.float32)
        mask = torch.ones((1, length, length), dtype=torch.bool, device=self.device)
        log_probs, pred_dist, pred_sep2, pred_rg_head, _ = self.model(emb, att, mask)
        probs = log_probs.exp()
        confidence = probs.max(dim=-1).values
        pred_dist_np = pred_dist[0, :length, :length].detach().cpu().numpy().astype(np.float32)
        confidence_np = confidence[0, :length, :length].detach().cpu().numpy().astype(np.float32)
        prob_tensor_np = probs[0, :length, :length, :].detach().cpu().numpy().astype(np.float32)
        implied_rg = np.sqrt(np.sum(pred_dist_np ** 2) / (2.0 * length * length))
        final_rg = float(implied_rg)
        pred_sep2_np = pred_sep2[0, :max(length - 2, 0)].detach().cpu().numpy().astype(np.float32)
        pred_rg_head_scalar = float(pred_rg_head.detach().cpu().reshape(-1)[0])
        return {'expected_dist': pred_dist_np, 'pair_confidence': confidence_np, 'prob_tensor': prob_tensor_np, 'pred_sep2': pred_sep2_np, 'pred_rg_head': pred_rg_head_scalar, 'pred_rg': final_rg, 'target_rg': final_rg, 'training_rgs': np.array([final_rg], dtype=np.float32), 'length': length}

class MartiniPhysicsProvider:

    def __init__(self, temperature: float=300):
        self.RT = 0.008314 * temperature
        self.aa_params = {'GLY': {'r0': 3.5, 'k': 700.0}, 'PRO': {'r0': 3.3, 'k': 1500.0}, 'DEFAULT': {'r0': 3.5, 'k': 1250.0}}

    def get_bond_params(self, res1: str, res2: str, interaction_type: str='1-2'):
        p1 = self.aa_params.get(res1, self.aa_params['DEFAULT'])
        p2 = self.aa_params.get(res2, self.aa_params['DEFAULT'])
        if interaction_type == '1-2':
            avg_r0 = (p1['r0'] + p2['r0']) / 2.0
            k_eff = p1['k'] * p2['k'] / (p1['k'] + p2['k']) * 2.0
            weight = k_eff / self.RT * 10.0
            return (avg_r0, weight)
        if interaction_type == '1-3':
            avg_r0 = 5.5
            k_eff = 500.0
            weight = k_eff / self.RT * 5.0
            return (avg_r0, weight)
        raise ValueError(f'Unknown interaction type: {interaction_type}')

class ProbabilisticSparseLaplacianIDP:

    def __init__(self, K: int=32, affinity_sigma: float=12.0, bond_length: float=3.8):
        self.K = K
        self.affinity_sigma = affinity_sigma
        self.bond_length = bond_length
        self.physics = MartiniPhysicsProvider()
        self.expected_dist = None
        self.pair_confidence = None
        self.training_rgs = None
        self.pred_rg = None
        self.target_rg = None
        self.chain_ids = None
        self.resnames = None
        self.resseqs = None
        self._cached_std_dev = None
        self._cached_exp_dist = None

    @classmethod
    def from_encoder_prediction(cls, pred_dict: dict, sequence: str, cfg: EncoderConfig):
        obj = cls(K=cfg.OUT_BINS, affinity_sigma=cfg.AFFINITY_SIGMA, bond_length=cfg.BOND_LENGTH)
        obj.expected_dist = np.asarray(pred_dict['expected_dist'], dtype=np.float32)
        obj.pair_confidence = np.asarray(pred_dict['pair_confidence'], dtype=np.float32)
        rg = float(pred_dict['target_rg'])
        obj.pred_rg = np.array(rg, dtype=np.float32)
        obj.target_rg = np.array(rg, dtype=np.float32)
        obj.training_rgs = np.array([rg], dtype=np.float32)
        length = len(sequence)
        obj.resnames = sequence_to_resnames(sequence)
        obj.chain_ids = np.array(['A'] * length)
        obj.resseqs = np.arange(1, length + 1)
        return obj

    def get_target_rg(self, default: float=30.0, apply_calibration: bool=True) -> float:
        rg = as_float_scalar(getattr(self, 'target_rg', None), default=None)
        if rg is None or not np.isfinite(rg) or rg <= 0:
            rg = as_float_scalar(getattr(self, 'pred_rg', None), default=None)
        if rg is None or not np.isfinite(rg) or rg <= 0:
            training_rgs = getattr(self, 'training_rgs', None)
            if training_rgs is not None:
                arr = np.asarray(training_rgs, dtype=np.float64).reshape(-1)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    rg = float(np.mean(arr))
        if rg is None or not np.isfinite(rg) or rg <= 0:
            rg = float(default)
        rg = float(rg)
        if apply_calibration and APPLY_RG_CALIBRATION:
            rg *= RG_MEAN_SCALE
        return rg

    def _calculate_structural_variability(self):
        length = self.expected_dist.shape[0]
        std_dev = np.zeros((length, length), dtype=np.float32)
        upper_indices = np.triu_indices(length, k=1)
        confidence = np.clip(self.pair_confidence[upper_indices], 0.01, 1.0)
        std_dev[upper_indices] = 15.0 * (1.0 - confidence)
        std_dev[upper_indices[1], upper_indices[0]] = std_dev[upper_indices]
        return (std_dev, self.expected_dist)

    def get_structural_variability_cached(self):
        if self._cached_std_dev is None:
            self._cached_std_dev, self._cached_exp_dist = self._calculate_structural_variability()
        return (self._cached_std_dev, self._cached_exp_dist)

    def _sample_targets_and_weights_hierarchical(self, rng: np.random.Generator, target_rg_val: float, sample_mix: float=DECODER_SAMPLE_MIX, lam: float=DECODER_LAM, p: float=DECODER_P, conf_floor: float=0.02, conf_phase_floor: float=0.1, hub_residue_percentile: float=90, hub_weight_scale: float=DECODER_HUB_WEIGHT_SCALE, weight_cap_quantile: float=99.5):
        length = self.expected_dist.shape[0]
        delta_matrix = np.zeros((length, length), dtype=np.float32)
        weight_matrix = np.zeros((length, length), dtype=np.float32)
        anchor_matrix = np.zeros((length, length), dtype=np.float32)
        upper_indices = np.triu_indices(length, k=1)
        std_dev, expected_dist = self.get_structural_variability_cached()
        pair_std = std_dev[upper_indices].astype(np.float32)
        pair_expected = expected_dist[upper_indices].astype(np.float32)
        raw_noise = rng.standard_normal((length, length))
        raw_noise = (raw_noise + raw_noise.T) / np.sqrt(2.0)
        blurred_noise = ndimage.gaussian_filter(raw_noise, sigma=max(5.0, length / 15.0))
        blurred_noise = (blurred_noise - blurred_noise.mean()) / (blurred_noise.std() + 1e-08)
        noise_1d = blurred_noise[upper_indices].astype(np.float32)
        sampled_dist = np.clip(pair_expected + pair_std * noise_1d, a_min=3.8, a_max=None)
        mixed_sample = (sample_mix * sampled_dist + (1.0 - sample_mix) * pair_expected).astype(np.float32)
        residue_score = np.median(std_dev, axis=1).astype(np.float32)
        residue_threshold = float(np.percentile(residue_score, hub_residue_percentile))
        hub_pair = (residue_score > residue_threshold)[upper_indices[0]] | (residue_score > residue_threshold)[upper_indices[1]]
        confidence = np.clip(self.pair_confidence[upper_indices], conf_floor, 1.0)
        confidence[hub_pair] *= 0.5
        delta = (confidence * pair_expected + (1.0 - confidence) * mixed_sample).astype(np.float32)
        sequence_separation = np.abs(upper_indices[0] - upper_indices[1]).astype(np.float32)
        base_flory = 3.8 * sequence_separation ** 0.588
        flory_matrix = np.zeros((length, length), dtype=np.float32)
        flory_matrix[upper_indices] = base_flory
        flory_matrix[upper_indices[1], upper_indices[0]] = base_flory
        flory_rg = np.sqrt(np.sum(flory_matrix ** 2) / (2.0 * length * length))
        ideal_idp_dist = base_flory * (target_rg_val / (flory_rg + 1e-08))
        base_weight = (np.exp(-(delta / self.affinity_sigma) ** 2) * (0.02 + self.pair_confidence[upper_indices])).astype(np.float32)
        weight = (base_weight * (1.0 + lam * confidence ** p)).astype(np.float32)
        weight[hub_pair] *= hub_weight_scale
        mask = (delta <= 20.0) & (self.pair_confidence[upper_indices] >= 0.15)
        masked_weight = weight[mask]
        if masked_weight.size > 0:
            cap = float(np.percentile(masked_weight, weight_cap_quantile))
            weight = np.minimum(weight, cap).astype(np.float32)
        soft_anchor = (conf_phase_floor + (1.0 - conf_phase_floor) * confidence).astype(np.float32)
        delta_matrix[upper_indices] = np.where(mask, delta, ideal_idp_dist)
        delta_matrix[upper_indices[1], upper_indices[0]] = delta_matrix[upper_indices]
        scaffold_weight = (0.01 * np.exp(-sequence_separation / (length * 0.4))).astype(np.float32)
        weight_matrix[upper_indices] = np.where(mask, weight, scaffold_weight).astype(np.float32)
        weight_matrix[upper_indices[1], upper_indices[0]] = weight_matrix[upper_indices]
        anchor_matrix[upper_indices] = np.where(mask, soft_anchor, 0.1).astype(np.float32)
        anchor_matrix[upper_indices[1], upper_indices[0]] = anchor_matrix[upper_indices]
        for index in range(length - 1):
            if self.chain_ids is not None and self.chain_ids[index] == self.chain_ids[index + 1]:
                r0_12, physics_weight_12 = self.physics.get_bond_params(self.resnames[index], self.resnames[index + 1], '1-2')
                delta_matrix[index, index + 1] = r0_12
                delta_matrix[index + 1, index] = r0_12
                weight_matrix[index, index + 1] = physics_weight_12
                weight_matrix[index + 1, index] = physics_weight_12
                anchor_matrix[index, index + 1] = 1.0
                anchor_matrix[index + 1, index] = 1.0
        for index in range(length - 2):
            if self.chain_ids is not None and self.chain_ids[index] == self.chain_ids[index + 2]:
                r0_13, physics_weight_13 = self.physics.get_bond_params(self.resnames[index], self.resnames[index + 2], '1-3')
                delta_matrix[index, index + 2] = r0_13
                delta_matrix[index + 2, index] = r0_13
                weight_matrix[index, index + 2] = max(weight_matrix[index, index + 2], physics_weight_13)
                weight_matrix[index + 2, index] = weight_matrix[index, index + 2]
        return (delta_matrix, weight_matrix, anchor_matrix)

def _smacof_core_loop(coordinates: torch.Tensor, target_distances: torch.Tensor, weights: torch.Tensor, laplacian_pinv: torch.Tensor, diagonal_indices: torch.Tensor, max_iter: int, tolerance: float) -> torch.Tensor:
    for _ in range(max_iter):
        distances = torch.cdist(coordinates, coordinates)
        safe_distances = torch.clamp(distances, min=1e-08)
        b_matrix = -(weights * target_distances / safe_distances)
        b_matrix[:, diagonal_indices, diagonal_indices] = 0.0
        b_matrix[:, diagonal_indices, diagonal_indices] = -b_matrix.sum(dim=2)
        new_coordinates = torch.bmm(laplacian_pinv, torch.bmm(b_matrix, coordinates))
        difference = torch.norm(new_coordinates - coordinates, dim=(1, 2)) / (torch.norm(coordinates, dim=(1, 2)) + 1e-08)
        coordinates = new_coordinates
        if torch.max(difference) < tolerance:
            break
    return coordinates

def generate_ensemble_smacof_hierarchical_gpu(delta_batch: np.ndarray, weight_batch: np.ndarray, anchor_mask_batch: np.ndarray, target_rgs: np.ndarray, bond_length: float, device: torch.device, divergency_mask: np.ndarray, max_iter: int=DEFAULT_MAX_ITER, tolerance: float=0.0001, rng_seed: int=DEFAULT_SEED) -> np.ndarray:
    batch_size, length, _ = delta_batch.shape
    generator = torch.Generator(device=device)
    generator.manual_seed(int(rng_seed))
    steps = torch.randn((batch_size, length, 3), dtype=torch.float64, device=device, generator=generator)
    steps = steps / (torch.norm(steps, dim=2, keepdim=True) + 1e-09) * bond_length
    coordinates = torch.cumsum(steps, dim=1)
    center = coordinates.mean(dim=1, keepdim=True)
    centered = coordinates - center
    current_rg = torch.sqrt(torch.mean(torch.sum(centered ** 2, dim=2), dim=1))
    target_rg_tensor = torch.tensor(target_rgs, dtype=torch.float64, device=device)
    coordinates = centered * (target_rg_tensor / (current_rg + 1e-08)).view(batch_size, 1, 1)
    diagonal_indices = torch.arange(length, device=device)
    target_distances = torch.tensor(delta_batch, dtype=torch.float64, device=device)
    full_weights = torch.tensor(weight_batch, dtype=torch.float64, device=device)
    anchor_mask = torch.tensor(anchor_mask_batch, dtype=torch.float64, device=device)
    phase_1_weights = full_weights * anchor_mask
    phase_1_laplacian = -phase_1_weights.clone()
    phase_1_laplacian[:, diagonal_indices, diagonal_indices] = 0.0
    phase_1_laplacian[:, diagonal_indices, diagonal_indices] = -phase_1_laplacian.sum(dim=2)
    identity = torch.eye(length, device=device, dtype=torch.float64).unsqueeze(0)
    phase_1_pinv = torch.linalg.pinv(phase_1_laplacian + identity * 1e-06)
    coordinates = _smacof_core_loop(coordinates=coordinates, target_distances=target_distances, weights=phase_1_weights, laplacian_pinv=phase_1_pinv, diagonal_indices=diagonal_indices, max_iter=int(max_iter * 0.6), tolerance=tolerance * 10)
    phase_2_laplacian = -full_weights.clone()
    phase_2_laplacian[:, diagonal_indices, diagonal_indices] = 0.0
    phase_2_laplacian[:, diagonal_indices, diagonal_indices] = -phase_2_laplacian.sum(dim=2)
    phase_2_pinv = torch.linalg.pinv(phase_2_laplacian + identity * 1e-08)
    coordinates = _smacof_core_loop(coordinates=coordinates, target_distances=target_distances, weights=full_weights, laplacian_pinv=phase_2_pinv, diagonal_indices=diagonal_indices, max_iter=max_iter, tolerance=tolerance)
    center = coordinates.mean(dim=1, keepdim=True)
    centered = coordinates - center
    squared_radius = torch.sum(centered ** 2, dim=2)
    divergency = torch.tensor(divergency_mask, dtype=torch.float64, device=device).unsqueeze(0)
    a_coefficient = torch.sum(squared_radius * divergency ** 2, dim=1)
    b_coefficient = 2.0 * torch.sum(squared_radius * divergency, dim=1)
    c_coefficient = torch.sum(squared_radius, dim=1) - length * target_rg_tensor ** 2
    discriminant = torch.clamp(b_coefficient ** 2 - 4.0 * a_coefficient * c_coefficient, min=0.0)
    alpha = (-b_coefficient + torch.sqrt(discriminant)) / (2.0 * a_coefficient + 1e-08)
    local_scale = torch.clamp(alpha.unsqueeze(1) * divergency, min=-0.5, max=3.0)
    scale_factor = 1.0 + local_scale
    scaled_coordinates = centered * scale_factor.unsqueeze(2)
    final_coordinates = _smacof_core_loop(coordinates=scaled_coordinates, target_distances=target_distances, weights=full_weights, laplacian_pinv=phase_2_pinv, diagonal_indices=diagonal_indices, max_iter=150, tolerance=tolerance * 10)
    if length >= 4:
        vector_1 = final_coordinates[:, 1:-2, :] - final_coordinates[:, 0:-3, :]
        vector_2 = final_coordinates[:, 2:-1, :] - final_coordinates[:, 1:-2, :]
        vector_3 = final_coordinates[:, 3:, :] - final_coordinates[:, 2:-1, :]
        cross_12 = torch.cross(vector_1, vector_2, dim=2)
        chiral_volume = torch.sum(cross_12 * vector_3, dim=2)
        net_chirality = torch.sum(chiral_volume, dim=1)
        is_mirror = net_chirality > 0
        z_multiplier = torch.where(is_mirror, torch.tensor(-1.0, device=device, dtype=torch.float64), torch.tensor(1.0, device=device, dtype=torch.float64))
        final_coordinates[:, :, 2] *= z_multiplier.unsqueeze(1)
    center = final_coordinates.mean(dim=1, keepdim=True)
    centered = final_coordinates - center
    final_rg = torch.sqrt(torch.mean(torch.sum(centered ** 2, dim=2), dim=1))
    final_coordinates = centered * (target_rg_tensor / (final_rg + 1e-08)).view(batch_size, 1, 1)
    return final_coordinates.float().cpu().numpy()

def save_raw_ca_ensemble(coordinates_ensemble: np.ndarray, model_info: ProbabilisticSparseLaplacianIDP, output_path: Path) -> None:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    structure = Structure.Structure('MANTA_Ensemble')
    for model_index, coordinates in enumerate(coordinates_ensemble):
        model = Model.Model(model_index)
        chain = Chain.Chain('A')
        for atom_index, (residue_name, residue_number, position) in enumerate(zip(model_info.resnames, model_info.resseqs, coordinates)):
            residue = Residue.Residue((' ', int(residue_number), ' '), str(residue_name), atom_index)
            atom = Atom.Atom('CA', position.tolist(), 0.0, 1.0, ' ', 'CA', atom_index, 'C')
            residue.add(atom)
            chain.add(residue)
        model.add(chain)
        structure.add(model)
    io.set_structure(structure)
    io.save(str(output_path))

def decode_to_pdb(
    pred_dict: dict,
    sequence: str,
    target_name: str,
    output_pdb: Path,
    device: torch.device,
    target_rg_override: float | None = None,
    num_frames: int = DEFAULT_NUM_FRAMES,
    rg_std_percent: float = DEFAULT_RG_STD_PERCENT,
) -> dict:
    if int(num_frames) <= 0:
        raise ValueError('num_frames must be a positive integer.')
    num_frames = int(num_frames)

    rg_std_percent = float(rg_std_percent)
    if not np.isfinite(rg_std_percent) or rg_std_percent < 0:
        raise ValueError('rg_std_percent must be a finite non-negative value.')
    rg_std_internal_scale = RG_STD_BASE_SCALE * (rg_std_percent / 100.0)

    total_start = time.perf_counter()
    cfg = EncoderConfig()
    model = ProbabilisticSparseLaplacianIDP.from_encoder_prediction(pred_dict=pred_dict, sequence=sequence, cfg=cfg)

    target_rg_override = as_float_scalar(target_rg_override, default=None)

    if target_rg_override is not None:
        if target_rg_override <= 0:
            raise ValueError('target_rg_override must be positive.')
        raw_target_rg = max(float(target_rg_override), 0.001)
        base_target_rg = raw_target_rg
        target_rg = raw_target_rg
        target_rg_source = 'external'
    else:
        raw_target_rg = max(float(model.get_target_rg(default=30.0, apply_calibration=False)), 0.001)
        base_target_rg = max(raw_target_rg * RG_BASE_CALIB_SCALE, 0.001)
        target_rg = max(raw_target_rg * RG_MEAN_SCALE, 0.001)
        target_rg_source = 'sequence_predicted'

    rg_std = base_target_rg * rg_std_internal_scale
    std_dev, _ = model.get_structural_variability_cached()
    residue_divergency = np.median(std_dev, axis=1)
    div_min = residue_divergency.min()
    div_max = residue_divergency.max()
    normalized_divergency = 0.2 + 0.8 * (residue_divergency - div_min) / (div_max - div_min + 1e-08)
    base_seed_offset = int(hashlib.md5(f'{target_name}_{DEFAULT_SEED}'.encode('utf-8')).hexdigest(), 16) % 2 ** 31
    valid_coordinates = []
    seed_offset = 0
    while len(valid_coordinates) < num_frames:
        needed = num_frames - len(valid_coordinates)
        batch_size = min(DEFAULT_DECODER_BATCH_SIZE, needed)
        delta_list = []
        weight_list = []
        anchor_list = []
        target_rg_list = []
        for _ in range(batch_size):
            rng = np.random.default_rng(base_seed_offset + seed_offset)
            seed_offset += 1
            frame_rg = np.clip(rng.normal(target_rg, rg_std), target_rg * 0.5, target_rg * 2.0)
            delta_matrix, weight_matrix, anchor_matrix = model._sample_targets_and_weights_hierarchical(rng=rng, target_rg_val=frame_rg, lam=DECODER_LAM, p=DECODER_P, hub_weight_scale=DECODER_HUB_WEIGHT_SCALE, sample_mix=DECODER_SAMPLE_MIX)
            target_rg_list.append(frame_rg)
            delta_list.append(delta_matrix)
            weight_list.append(weight_matrix)
            anchor_list.append(anchor_matrix)
        coordinates_batch = generate_ensemble_smacof_hierarchical_gpu(delta_batch=np.asarray(delta_list), weight_batch=np.asarray(weight_list), anchor_mask_batch=np.asarray(anchor_list), target_rgs=np.asarray(target_rg_list, dtype=np.float32), bond_length=model.bond_length, device=device, divergency_mask=normalized_divergency, max_iter=DEFAULT_MAX_ITER, rng_seed=base_seed_offset + seed_offset)
        valid_coordinates.extend(coordinates_batch)
        del coordinates_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    raw_coordinates = np.asarray(valid_coordinates[:num_frames], dtype=np.float32)
    save_raw_ca_ensemble(coordinates_ensemble=raw_coordinates, model_info=model, output_path=output_pdb)
    return {
        'target_rg_source': target_rg_source,
        'target_rg_override': target_rg_override,
        'raw_target_rg': raw_target_rg,
        'target_mean_rg': target_rg,
        'target_rg_std': rg_std,
        'rg_std_percent': rg_std_percent,
        'rg_std_internal_scale': rg_std_internal_scale,
        'num_frames': num_frames,
        'decoder_and_pdb_sec': time.perf_counter() - total_start,
    }

class SequenceToPDBPipeline:

    def __init__(self, device: torch.device):
        self.cfg = EncoderConfig()
        self.device = torch.device(device)
        self.extractor = DirectESM2Extractor(model_name=DEFAULT_ESM_MODEL, device=self.device)
        self.encoder = DirectEncoderPredictor(cfg=self.cfg, checkpoint_path=DEFAULT_ENCODER_CHECKPOINT, device=self.device)

    def generate(
        self,
        sequence: str,
        output_pdb: Path,
        target_rg_override: float | None = None,
        num_frames: int = DEFAULT_NUM_FRAMES,
        rg_std_percent: float = DEFAULT_RG_STD_PERCENT,
    ) -> dict:
        sequence = re.sub('\\s+', '', str(sequence)).upper()
        output_pdb = Path(output_pdb).expanduser().resolve()
        if not sequence:
            raise ValueError('The sequence is empty.')
        if len(sequence) > self.cfg.MAX_LEN:
            raise ValueError(f'Sequence length {len(sequence)} exceeds MAX_LEN={self.cfg.MAX_LEN}.')
        if not re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+', sequence):
            raise ValueError('The sequence contains non-canonical amino acids. Only the 20 standard one-letter residue codes are accepted.')
        output_pdb.parent.mkdir(parents=True, exist_ok=True)
        target_name = output_pdb.stem
        total_start = time.perf_counter()
        esm_start = time.perf_counter()
        embeddings, attentions = self.extractor.extract(sequence)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        esm_seconds = time.perf_counter() - esm_start
        encoder_start = time.perf_counter()
        prediction = self.encoder.predict(embeddings, attentions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        encoder_seconds = time.perf_counter() - encoder_start
        del embeddings
        del attentions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        decoder_info = decode_to_pdb(
            pred_dict=prediction,
            sequence=sequence,
            target_name=target_name,
            output_pdb=output_pdb,
            device=self.device,
            target_rg_override=target_rg_override,
            num_frames=num_frames,
            rg_std_percent=rg_std_percent,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result = {'output_pdb': str(output_pdb), 'sequence_length': len(sequence), 'esm2_sec': esm_seconds, 'encoder_sec': encoder_seconds, **decoder_info, 'end_to_end_sec': time.perf_counter() - total_start}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a Cα-level IDP ensemble PDB from one amino-acid sequence.')
    parser.add_argument('sequence', type=str, help='Protein sequence using the 20 standard one-letter amino-acid codes.')
    parser.add_argument('output_pdb', type=Path, help='Output multi-model PDB path.')
    parser.add_argument(
        '--target-rg',
        type=float,
        default=None,
        help='External target Rg in Angstrom. If omitted, the sequence-predicted Rg is used.',
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f'Number of conformers to generate. Default: {DEFAULT_NUM_FRAMES}.',
    )
    parser.add_argument(
        '--rg-std-scale', '--rg-std-percent',
        dest='rg_std_percent',
        type=float,
        default=DEFAULT_RG_STD_PERCENT,
        help=(
            'Relative Rg-distribution width in percent. '
            '100 corresponds to the paper/default setting (internal RG_STD scale = 0.30); '
            '50 gives 0.15 and 200 gives 0.60. Default: 100.'
        ),
    )
    return parser.parse_args()

def print_summary(result: dict) -> None:
    print('\n' + '=' * 72)
    print('MANTA generation completed')
    print('=' * 72)
    print(f"Output PDB:       {result['output_pdb']}")
    print(f"Sequence length:  {result['sequence_length']}")
    print(f"Frames:           {result['num_frames']}")
    print(f"Target Rg source: {result['target_rg_source']}")
    print(f"Target mean Rg:   {result['target_mean_rg']:.3f} Å")
    print(f"Rg std width:     {result['rg_std_percent']:.1f}%")
    print(f"Internal std scale:{result['rg_std_internal_scale']:.3f}")
    print(f"Target Rg std:    {result['target_rg_std']:.3f} Å")
    print(f"ESM-2:            {result['esm2_sec']:.3f} sec")
    print(f"Encoder:          {result['encoder_sec']:.3f} sec")
    print(f"Decoder + PDB:    {result['decoder_and_pdb_sec']:.3f} sec")
    print(f"End-to-end:       {result['end_to_end_sec']:.3f} sec")

def main() -> None:
    args = parse_arguments()

    if args.num_frames <= 0:
        raise ValueError('--num-frames must be a positive integer.')
    if not np.isfinite(args.rg_std_percent) or args.rg_std_percent < 0:
        raise ValueError('--rg-std-scale/--rg-std-percent must be a finite non-negative percentage.')

    internal_rg_std_scale = RG_STD_BASE_SCALE * (args.rg_std_percent / 100.0)
    print(f'Conformers to generate: {args.num_frames}')
    print(f'Rg std width: {args.rg_std_percent:.1f}% (internal scale = {internal_rg_std_scale:.3f})')

    if args.target_rg is not None:
        if args.target_rg <= 0:
            raise ValueError('--target-rg must be positive.')

        print('\n' + '!' * 72)
        print('WARNING: EXTERNAL Rg OVERRIDE ENABLED')
        print('!' * 72)
        print(f'External target Rg: {args.target_rg:.3f} Å')
        print(
            'An excessively small or large external Rg can force the generated '
            'ensemble into strongly over-compact or over-extended conformations '
            'and may produce distorted or nonphysical geometry.'
        )
        print(
            'Use external Rg values only when the requested compactness is '
            'physically meaningful for the target system.'
        )
        print('!' * 72)
        input('Press Enter to continue, or Ctrl+C to cancel... ')

    seed_everything(DEFAULT_SEED)
    device = resolve_device()
    pipeline = SequenceToPDBPipeline(device=device)
    result = pipeline.generate(
        sequence=args.sequence,
        output_pdb=args.output_pdb,
        target_rg_override=args.target_rg,
        num_frames=args.num_frames,
        rg_std_percent=args.rg_std_percent,
    )
    print_summary(result)
if __name__ == '__main__':
    main()
