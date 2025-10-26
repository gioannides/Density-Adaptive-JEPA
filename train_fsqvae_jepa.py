#!/usr/bin/env python
# train_jepa_fsqvae_hifigan.py
# Stage 1: JEPA self-supervised encoder training
# Stage 2: Frozen encoder + FSQ-VAE + HiFi-GAN decoder training

import os, json, argparse, random, math, time
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from collections import deque

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import soundfile
import deepspeed

# ------------------------------
# Utilities
# ------------------------------

def print_model_stats(model, stage_name="Model"):
    """Print model parameter counts in a formatted table"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("\n" + "="*80)
    print(f"{stage_name} Architecture Summary")
    print("="*80)
    print(f"{'Component':<25} {'Parameters':<15} {'Trainable':<15}")
    print("-"*80)
    
    # Component breakdown
    components = {}
    for name, module in model.named_children():
        comp_params = sum(p.numel() for p in module.parameters())
        comp_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        components[name] = (comp_params, comp_trainable)
    
    for name, (params, trainable) in sorted(components.items(), key=lambda x: -x[1][0]):
        print(f"{name:<25} {params/1e6:>6.2f}M ({params/total_params*100:>4.1f}%)   {trainable/1e6:>6.2f}M")
    
    print("-"*80)
    print(f"{'TOTAL':<25} {total_params/1e6:>6.2f}M           {trainable_params/1e6:>6.2f}M")
    if frozen_params > 0:
        print(f"{'Frozen (lr=0)':<25} {frozen_params/1e6:>6.2f}M")
    print("="*80 + "\n")

@torch.no_grad()
def _fsq_dim_radices(D: int, levels: List[int], device=None) -> torch.Tensor:
    assert D % len(levels) == 0, f"D={D} must be divisible by len(levels)={len(levels)}"
    per = D // len(levels)
    r = []
    for L in levels:
        r += [int(L)] * per
    return torch.tensor(r, dtype=torch.long, device=device)

@torch.no_grad()
def fsq_pack_indices(indices: torch.Tensor, levels: List[int], group_size: int = 7) -> torch.Tensor:
    B, T, D = indices.shape
    device = indices.device
    rad = _fsq_dim_radices(D, levels, device=device)
    G = (D + group_size - 1) // group_size

    pad = G * group_size - D
    if pad > 0:
        indices = torch.cat([indices, torch.zeros(B, T, pad, dtype=indices.dtype, device=device)], dim=2)
        rad = torch.cat([rad, torch.ones(pad, dtype=rad.dtype, device=device)], dim=0)

    toks = torch.zeros(B, T, 0, dtype=torch.long, device=device)
    for g in range(G):
        s, e = g * group_size, (g + 1) * group_size
        chunk = indices[:, :, s:e].long()
        rchunk = rad[s:e].long()
        tok = torch.zeros(B, T, dtype=torch.long, device=device)
        for k in range(rchunk.numel() - 1, -1, -1):
            tok = chunk[:, :, k] + tok * rchunk[k]
        toks = torch.cat([toks, tok.unsqueeze(-1)], dim=-1)
    return toks

@torch.no_grad()
def fsq_token_stats_from_indices(
    indices: torch.Tensor,
    fsq_levels: List[int],
    code_dim: int,
    sample_rate: int,
    strides: List[int],
    group_size: int = 7
) -> dict:
    hop = int(np.prod(strides))
    fps = float(sample_rate) / float(hop)
    packed = fsq_pack_indices(indices, levels=fsq_levels, group_size=group_size)
    B, T, G = packed.shape
    tokens_total = int(T) * int(G)
    tokens_per_sec = fps * float(G)
    seconds = float(T) / fps if fps > 0.0 else float("inf")
    return {
        "B": int(B), "T": int(T), "G": int(G), "fps": float(fps),
        "seconds": float(seconds), "tokens_total": int(tokens_total),
        "tokens_per_sec": float(tokens_per_sec), "group_size": int(group_size),
        "code_dim": int(code_dim), "hop": int(hop),
    }

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def rank0() -> bool:
    try:
        return (not is_distributed()) or (dist.get_rank() == 0)
    except Exception:
        return True

def load_mono_resample(path, target_sr=24000, clean: str | bool = False):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

# ------------------------------
# JEPA Masking Strategy
# ------------------------------

def create_jepa_mask(batch_size: int, seq_len: int, mask_ratio: float = 0.5, 
                     min_span: int = 4, max_span: int = 16, device='cuda'):
    """
    Create JEPA-style block masks for temporal sequences.
    Returns: mask [B, T] where 1=keep, 0=mask
    """
    masks = torch.ones(batch_size, seq_len, device=device)
    
    for b in range(batch_size):
        num_to_mask = int(seq_len * mask_ratio)
        masked_so_far = 0
        
        while masked_so_far < num_to_mask:
            span_len = random.randint(min_span, max_span)
            start = random.randint(0, max(1, seq_len - span_len))
            end = min(start + span_len, seq_len)
            
            masks[b, start:end] = 0
            masked_so_far += (end - start)
            
            if masked_so_far >= num_to_mask:
                break
    
    return masks

# ------------------------------
# Gaussian Adaptive Attention
# ------------------------------

class GaussianAdaptiveAttention(nn.Module):
    def __init__(self, norm_axis: int, num_heads: int, num_gaussians: int,
                 padding_value=None, mean_offset_init: float = 0.0, eps: float = 1e-8):
        super().__init__()
        if not isinstance(norm_axis, int):
            raise ValueError("norm_axis must be an integer.")
        if num_heads <= 0:
            raise ValueError("num_heads must be a positive integer.")
        if num_gaussians <= 0:
            raise ValueError("num_gaussians must be a positive integer.")

        self.norm_axis = norm_axis
        self.num_heads = num_heads
        self.num_gaussians = num_gaussians
        self.padding_value = padding_value
        self.eps = eps

        self.mean_offsets = nn.Parameter(torch.full((num_gaussians,), float(mean_offset_init)))
        self.log_sigma = nn.Parameter(torch.full((num_gaussians,), math.log(0.5)))
        self.register_buffer("_log_sqrt_2pi", torch.tensor(0.5 * math.log(2.0 * math.pi)), persistent=False)

    def forward(self, x, return_attention_details: bool = False):
        with torch.cuda.amp.autocast(enabled=False):
            xf = x.float()
            mean_offsets = self.mean_offsets.float()
            log_sigma = self.log_sigma.float()

            mean = xf.mean(dim=self.norm_axis, keepdim=True)
            var = xf.var(dim=self.norm_axis, keepdim=True, unbiased=False)
            std = var.clamp_min(1e-6).sqrt()
            sigma = F.softplus(self.log_sigma) + 1e-3
            
            log_terms = []
            for k in range(self.num_gaussians):
                z = (xf - (mean + mean_offsets[k])) / (std * sigma[k] + 1e-8)
                log_terms.append(-0.5*(z*z) - torch.log(sigma[k]) - torch.tensor(0.5*math.log(2*math.pi)))
            
            log_G = torch.stack(log_terms, dim=-1)
            log_gate = torch.logsumexp(log_G, dim=-1) - math.log(self.num_gaussians)
            gate32 = torch.exp(log_gate)
            out32 = xf * gate32
        
        out = out32.to(x.dtype)
        if return_attention_details: 
            return out, gate32.to(x.dtype)
        return out

class GAttnGateG(nn.Module):
    def __init__(self, in_ch, num_gaussians=4, cap=0.2):
        super().__init__()
        self.to_attn = nn.Conv1d(in_ch, 1, 1)
        self.gaa = GaussianAdaptiveAttention(norm_axis=2, num_heads=1, num_gaussians=num_gaussians)
        self.alpha = nn.Parameter(torch.tensor(0.05))
        self.cap = cap

    def forward(self, x):
        a = self.to_attn(x)
        _, gate = self.gaa(a, return_attention_details=True)
        alpha = self.alpha
        scale = 1.0 + alpha * gate
        y = x * scale
        return y, gate

# ------------------------------
# MR-STFT Loss
# ------------------------------

class MRSTFTLoss(nn.Module):
    def __init__(self, 
                 fft_sizes=[2048, 1024, 512, 256, 128],
                 hop_sizes=[512, 256, 128, 64, 32],
                 win_lengths=[2048, 1024, 512, 256, 128],
                 mag_weight=1.0,
                 log_mag_weight=1.0):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.mag_weight = mag_weight
        self.log_mag_weight = log_mag_weight
        for w in win_lengths:
            self.register_buffer(f"window_{w}", torch.hann_window(w), persistent=False)

    def stft(self, x, fft_size, hop_size, win_length):
        window = getattr(self, f"window_{win_length}")
        x32 = x.float()
        w32 = window.to(device=x.device, dtype=torch.float32)
        return torch.stft(
            x32, n_fft=fft_size, hop_length=hop_size, win_length=win_length,
            window=w32, return_complex=True
        )
    
    def forward(self, pred, target, lengths=None):
        if lengths is not None:
            B = pred.shape[0]
            tot = 0.0
            for b in range(B):
                L = lengths[b]
                p = pred[b:b+1, :, :L]
                t = target[b:b+1, :, :L]
                ssum = 0.0
                used = 0
                for n,h,w in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
                    if L < n: continue
                    used += 1
                    ps = self.stft(p.squeeze(1), n,h,w)
                    ts = self.stft(t.squeeze(1), n,h,w)
                    pm, tm = ps.abs(), ts.abs()
                    ssum += self.mag_weight*F.l1_loss(pm,tm) + self.log_mag_weight*F.l1_loss((pm+1e-5).log(), (tm+1e-5).log())
                tot += ssum/max(1,used)
            return tot/B

        loss = 0.0
        for n,h,w in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            ps = self.stft(pred.squeeze(1),n,h,w)
            ts = self.stft(target.squeeze(1),n,h,w)
            pm, tm = ps.abs(), ts.abs()
            loss += self.mag_weight*F.l1_loss(pm,tm)
            loss += self.log_mag_weight*F.l1_loss((pm+1e-5).log(), (tm+1e-5).log())
        return loss/len(self.fft_sizes)

# ------------------------------
# Dataset
# ------------------------------

class StreamingWaveformDataset(IterableDataset):
    def __init__(self,
                 root_dir: str,
                 sample_rate: int = 24000,
                 max_seconds: float = 10.0,
                 sleep: float = 5.0,
                 rank: int = 0,
                 world_size: int = 1,
                 pattern: str = "*.jsonl",
                 json_audio_key: str = "wav_path",
                 augment: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.sleep = sleep
        self.rank = rank
        self.world_size = world_size
        self.pattern = pattern
        self.json_audio_key = json_audio_key
        self.augment = augment
        self.processed_file = os.path.join("./", 'jepa_audio_nogaatn.txt')
        self.processed = set()
        if os.path.exists(self.processed_file):
            self.processed = set(open(self.processed_file).read().splitlines())
            if self.rank == 0:
                print(f"Resuming: Skipping {len(self.processed)} already processed files")

    def _process_line(self, line: str):
        obj = json.loads(line)
        path = obj[self.json_audio_key]
        if self.rank == 0 and path in self.processed:
            return None
        wav, _ = load_mono_resample(path, target_sr=self.sample_rate)
        if self.rank == 0:
            with open(self.processed_file, 'a', buffering=1) as f:
                f.write(path + '\n')
                f.flush()
                os.fsync(f.fileno())
            self.processed.add(path)
        if self.max_seconds:
            max_samples = int(self.sample_rate * self.max_seconds)
            if wav.shape[-1] > max_samples:
                start = random.randint(0, wav.shape[-1] - max_samples)
                wav = wav[..., start:start+max_samples]
        if self.augment:
            if random.random() < 0.3:
                wav = wav * random.uniform(0.8, 1.2)
            if random.random() < 0.2:
                wav = (wav + torch.randn_like(wav) * 0.001).clamp_(-1, 1)
        return wav.squeeze(0)

    def _file_iter(self, fpath: str):
        with open(fpath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if (i % self.world_size) != self.rank:
                    continue
                line = line.strip()
                if not line: continue
                try:
                    out = self._process_line(line)
                    if out is not None: yield out
                except Exception as e:
                    print(f"Error processing line {i} in {fpath}: {e}")

    def __iter__(self):
        import glob
        seen = set()
        while True:
            files = sorted(glob.glob(os.path.join(self.root_dir, self.pattern)))
            new_files = [fp for fp in files if fp not in seen]
            if not new_files:
                time.sleep(self.sleep)
                continue
            for fp in new_files:
                seen.add(fp)
                yield from self._file_iter(fp)

# ------------------------------
# FSQ Quantizer
# ------------------------------

class FiniteScalarQuantizer(nn.Module):
    def __init__(self, 
                 levels: List[int], 
                 dim: int,
                 normalized: bool = True,
                 use_tanh: bool = True,
                 temperature: float = 1.0):
        super().__init__()
        assert dim % len(levels) == 0
        self.levels = levels
        self.dim = dim
        self.normalized = normalized
        self.use_tanh = use_tanh
        self.temperature = temperature
        self.dims_per_level = dim // len(levels)
        self.boundaries = nn.ModuleList()
        for L in levels:
            if normalized:
                bounds = torch.linspace(-1 + 1/L, 1 - 1/L, L)
            else:
                bounds = torch.linspace(1/(2*L), 1 - 1/(2*L), L)
            mod = nn.Module()
            mod.register_buffer('bounds', bounds)
            self.boundaries.append(mod)
        self.implicit_codebook_size = math.prod(levels)
    
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D, T = z.shape
        assert D == self.dim
        z = z.permute(0,2,1).contiguous()
        z_flat = z.view(-1, D)
        if self.use_tanh:
            z_flat = torch.tanh(z_flat / self.temperature) * self.temperature
        z_q_list, indices_list = [], []
        for i, L in enumerate(self.levels):
            s = i * self.dims_per_level
            e = s + self.dims_per_level
            z_group = z_flat[:, s:e]
            bounds = self.boundaries[i].bounds
            dist = (z_group.unsqueeze(-1) - bounds.view(1,1,L)).abs()
            idx = torch.argmin(dist, dim=-1)
            z_q_group = bounds[idx]
            z_q_list.append(z_q_group)
            indices_list.append(idx)
        z_q_flat = torch.cat(z_q_list, dim=1)
        all_idx = torch.cat(indices_list, dim=1)
        z_q = z_q_flat.view(B, -1, D).permute(0,2,1).contiguous()
        return z_q, all_idx.view(B, -1, D)
    
    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_q, indices = self.quantize(z_e)
        z_q = z_e + (z_q - z_e).detach()
        aux_loss = torch.tensor(0.0, device=z_e.device, dtype=z_e.dtype)
        return z_q, indices, aux_loss

# ------------------------------
# HiFi-GAN Components
# ------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=(kernel_size*d-d)//2)
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=(kernel_size-1)//2)
            for _ in dilation
        ])
    
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

class MRFBlock(nn.Module):
    def __init__(self, channels, kernels=[3,7,11], dilations=[(1,3,5),(1,3,5),(1,3,5)]):
        super().__init__()
        self.resblocks = nn.ModuleList([ResBlock(channels, k, d) for k,d in zip(kernels,dilations)])
    
    def forward(self, x):
        out = self.resblocks[0](x)
        for b in self.resblocks[1:]:
            out = out + b(x)
        return out/len(self.resblocks)

class SnakeBeta(nn.Module):
    def __init__(self, in_features, min_alpha=1e-2, max_inv=10.0):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(1, in_features, 1))
        self.min_alpha = min_alpha
        self.max_inv = max_inv
    
    def forward(self, x):
        alpha = F.softplus(self.raw) + self.min_alpha
        inv = (1.0 / alpha).clamp_max(self.max_inv)
        return x + inv * (torch.sin(alpha * x) ** 2)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_res=2, use_gaatn=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=2*stride,
                              stride=stride, padding=stride//2)
        self.res_blocks = nn.ModuleList([
            ResBlock(out_channels, kernel_size=3, dilation=(1, 3**i, 5**i))
            for i in range(n_res)
        ])
        self.snake = SnakeBeta(out_channels)
        self.use_gaatn = use_gaatn
        if use_gaatn:
            self.gaatn_gate = GAttnGateG(in_ch=out_channels, num_gaussians=4)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.snake(x)
        for b in self.res_blocks: 
            x = b(x)
        if self.use_gaatn:
            x, gate = self.gaatn_gate(x)
        else:
            gate = None
        return x, gate

class HiFiDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernels=[3,7,11,15,23,32], use_gaatn=True):
        super().__init__()
        self.snake = SnakeBeta(in_channels)
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2*stride,
                                         stride=stride, padding=stride//2)
        self.mrf = MRFBlock(out_channels, kernels)
        self.use_gaatn = use_gaatn
        if use_gaatn:
            self.gaatn_gate = GAttnGateG(in_ch=out_channels, num_gaussians=4)
    
    def forward(self, x):
        x = self.snake(x)
        x = self.deconv(x)
        x = self.mrf(x)
        if self.use_gaatn:
            x, gate = self.gaatn_gate(x)
        else:
            gate = None
        return x, gate



class ConformerBlock(nn.Module):
    def __init__(self, dim, heads=8, ff_mult=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        assert dim % heads == 0, f"embed_dim ({dim}) must be divisible by heads ({heads})"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.attn_drop_p = dropout

        # Feedforward 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )

        # Attention (SDPA path)
        self.norm_attn = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        # Conv module
        self.conv = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Conv1d(dim, 2 * dim, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=dim),
            nn.GroupNorm(1, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

        # Feedforward 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )

        self.norm_final = nn.LayerNorm(dim)

    def _shape_qkv(self, x_bt_d):
        # x_bt_d: [B, T, D]
        B, T, D = x_bt_d.shape
        qkv = self.qkv(x_bt_d)                              # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)                      # each [B, T, D]
        # -> [B, heads, T, head_dim]
        def split_heads(t):
            return t.view(B, T, self.heads, self.head_dim).transpose(1, 2).contiguous()
        return split_heads(q), split_heads(k), split_heads(v)

    def _merge_heads(self, x_bhtd):
        # x_bhtd: [B, H, T, Hd] -> [B, T, D]
        B, H, T, Hd = x_bhtd.shape
        return x_bhtd.transpose(1, 2).contiguous().view(B, T, H * Hd)

    def forward(self, x):
        # x: [B, D, T]
        B, D, T = x.shape

        # To [B, T, D]
        s = x.transpose(1, 2).contiguous()

        # FFN 1
        s = s + 0.5 * self.ff1(s)  # [B, T, D]

        # SDPA
        s_norm = self.norm_attn(s)  # [B, T, D]
        q, k, v = self._shape_qkv(s_norm)  # [B, H, T, Hd] each

        # Use PyTorch SDPA; dtype stays consistent with module params under ZeRO
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
        )  # [B, H, T, Hd]

        attn = self._merge_heads(attn)          # [B, T, D]
        attn = self.out_proj(attn)              # [B, T, D]
        s = s + attn                            # [B, T, D]

        # Conv branch on [B, D, T]
        c = s.transpose(1, 2)                   # [B, D, T]
        c = c + self.conv(c)                    # [B, D, T]

        # FFN 2
        s = c.transpose(1, 2)                   # [B, T, D]
        s = s + 0.5 * self.ff2(s)               # [B, T, D]

        # Final norm and back to [B, D, T]
        s = self.norm_final(s)                  # [B, T, D]
        return s.transpose(1, 2).contiguous()   # [B, D, T]



# ------------------------------
# JEPA Encoder (Stage 1)
# ------------------------------


def _conv1d_out_len(L, k, s, p, d=1):
    # PyTorch formula: floor((L + 2p - d*(k-1) - 1)/s + 1)
    return (L + 2*p - d*(k-1) - 1) // s + 1

def jepa_time_len_from_wav(T_wav: int, strides: List[int]) -> int:
    # input_conv: k=7, s=1, p=3  -> preserves length
    L = _conv1d_out_len(T_wav, k=7, s=1, p=3)
    for s in strides:
        k = 2 * s
        p = s // 2
        L = _conv1d_out_len(L, k=k, s=s, p=p)
    return L


class JEPAEncoder(nn.Module):
    """JEPA-style encoder for self-supervised representation learning"""
    def __init__(self,
                 sample_rate: int = 24000,
                 code_dim: int = 128,
                 channels: List[int] = [32, 64, 128, 256, 512],
                 strides: List[int] = [4, 4, 5, 4, 4],
                 n_res_blocks: int = 2,
                 n_conformer: int = 2,
                 conformer_heads: int = 4,
                 use_gaatn: bool = True,
                 debug: bool = False):
        super().__init__()
        assert len(channels) == len(strides) + 1
        self.sample_rate = sample_rate
        self.strides = strides
        self.hop_length = math.prod(strides)
        self.debug = debug
        
        # Context encoder (online, trainable)
        self.input_conv = nn.Conv1d(1, channels[0], kernel_size=7, padding=3)
        self.encoder = nn.ModuleList([
            EncoderBlock(channels[i], channels[i+1], strides[i], n_res_blocks, use_gaatn)
            for i in range(len(strides))
        ])
        
        self.bottleneck_proj = nn.Conv1d(channels[-1], code_dim, 1)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(code_dim, heads=conformer_heads, dropout=0.1) 
            for _ in range(n_conformer)
        ])
        
        # ✅ Add learnable mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, code_dim, 1))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # ✅ Predictor
        self.predictor = nn.Sequential(
            nn.Conv1d(code_dim, code_dim * 2, 1),
            nn.GELU(),
            ConformerBlock(code_dim * 2, heads=conformer_heads, dropout=0.1),
            nn.Conv1d(code_dim * 2, code_dim * 2, 1),
            nn.GELU(),
            ConformerBlock(code_dim * 2, heads=conformer_heads, dropout=0.1),
            nn.Conv1d(code_dim * 2, code_dim, 1)
        )
        
        self.code_dim = code_dim
        self.apply(self._init_weights)

        import copy

        # ✅ Create EMA target encoder as ModuleDict
        self.ema_decay = 0.996
        self.target_encoder = nn.ModuleDict({
            'input_conv': copy.deepcopy(self.input_conv),
            'encoder': copy.deepcopy(self.encoder),
            'bottleneck_proj': copy.deepcopy(self.bottleneck_proj),
            'conformer_blocks': copy.deepcopy(self.conformer_blocks)
        })
        
        # Freeze target encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: 
                nn.init.zeros_(m.bias)

    # ✅ CORRECT: Only ONE _target_encode method with dictionary access
    @torch.no_grad()
    def _target_encode(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.target_encoder['input_conv'](wav)
        for enc in self.target_encoder['encoder']:
            x, _ = enc(x)
        z = self.target_encoder['bottleneck_proj'](x)
        for conf in self.target_encoder['conformer_blocks']:
            if z.shape[-1] < 2:
                break
            z = conf(z)
        return z

    @torch.no_grad()
    def update_target_encoder(self, decay: Optional[float] = None):
        d = self.ema_decay if decay is None else decay

        def ema_update(tgt_mod, src_mod):
            for (_, p_t), (_, p_s) in zip(tgt_mod.named_parameters(), src_mod.named_parameters()):
                p_t.data.mul_(d).add_(p_s.data, alpha=1.0 - d)
            for (_, b_t), (_, b_s) in zip(tgt_mod.named_buffers(), src_mod.named_buffers()):
                b_t.data.copy_(b_s.data)

        ema_update(self.target_encoder['input_conv'],      self.input_conv)
        ema_update(self.target_encoder['encoder'],         self.encoder)
        ema_update(self.target_encoder['bottleneck_proj'], self.bottleneck_proj)
        for tb, sb in zip(self.target_encoder['conformer_blocks'], self.conformer_blocks):
            ema_update(tb, sb)
    
    def encode(self, wav: torch.Tensor):
        """Encode waveform to representations (online encoder)"""
        B, C, T_wav = wav.shape
        
        if self.debug:
            print(f"[ENCODER DEBUG] Input wav: {wav.shape}")
        
        x = self.input_conv(wav)
        if self.debug:
            print(f"[ENCODER DEBUG] After input_conv: {x.shape}")
        
        for i, enc in enumerate(self.encoder):
            x, _ = enc(x)
            T_curr = x.shape[-1]
            if self.debug:
                print(f"[ENCODER DEBUG] After encoder block {i} (stride={self.strides[i]}): {x.shape}")
            
            if T_curr < 2:
                raise RuntimeError(
                    f"Encoder block {i}: time dimension collapsed to {T_curr}. "
                    f"Input was [B={B}, C={C}, T={T_wav}]. "
                    f"Total stride reduction so far: {math.prod(self.strides[:i+1])}x. "
                    f"SOLUTION: Use gentler strides or increase audio length."
                )
        
        z = self.bottleneck_proj(x)
        if self.debug:
            print(f"[ENCODER DEBUG] After bottleneck_proj: {z.shape}")
        
        T_z = z.shape[-1]
        if T_z < 1:
            raise RuntimeError(
                f"Bottleneck output has T=0! "
                f"Your strides are too aggressive for the input length."
            )
        
        for i, conf in enumerate(self.conformer_blocks):
            T_before = z.shape[-1]
            
            if T_before < 2:
                if self.debug:
                    print(f"[ENCODER DEBUG] Skipping ConformerBlock {i} due to T={T_before} < 2")
                continue
            
            z = conf(z)
            T_after = z.shape[-1]
            
            if self.debug:
                print(f"[ENCODER DEBUG] After conformer block {i}: {z.shape}")
            
            if T_before != T_after:
                raise RuntimeError(f"ConformerBlock {i} changed time dimension: {T_before} -> {T_after}")
        
        if self.debug:
            print(f"[ENCODER DEBUG] Final output: {z.shape}, T={z.shape[-1]}")
        
        return z
    
    def forward(self, wav: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        JEPA forward pass
        
        Args:
            wav: [B, 1, T_wav] input audio
            mask: [B, T_z] binary mask (1=visible, 0=masked)
        
        Returns:
            z_context: [B, C, T_z] context features
            z_pred: [B, C, T_z] predicted features
            mask: [B, T_z] the mask
            z_target: [B, C, T_z] target features from EMA encoder
        """
        if mask is None:
            z_context = self.encode(wav)
            return z_context, None, None, None

        # ✅ CORRECT: Both encoders see FULL audio
        z_context = self.encode(wav)  # Online encoder
        
        # Target encoder (EMA, frozen)
        with torch.no_grad():
            z_target = self._target_encode(wav)  # Uses the CORRECT method

        B, C, Tz = z_target.shape
        if mask.shape != (B, Tz):
            raise ValueError(f"Mask shape mismatch: expected {(B, Tz)}, got {mask.shape}")

        # ✅ Mask at feature level
        mask_3d = mask.unsqueeze(1).to(device=z_context.device, dtype=z_context.dtype)
        mask_tokens = self.mask_token.expand(B, -1, Tz).to(device=z_context.device, dtype=z_context.dtype)
        
        # Create masked context: visible=context, hidden=mask_token
        z_masked = z_context * mask_3d + mask_tokens * (1 - mask_3d)
        
        # Predict targets from masked context
        z_pred = self.predictor(z_masked)

        return z_context, z_pred, mask, z_target

# ------------------------------
# Full Model (Stage 2: Frozen Encoder + FSQ + Decoder)
# ------------------------------

class WaveformJEPAFSQVAE(nn.Module):
    def __init__(self,
                 jepa_encoder: Optional[JEPAEncoder] = None,
                 fsq_levels: List[int] = [8, 8, 8, 8],
                 channels: List[int] = [32, 64, 128, 256, 512],
                 strides: List[int] = [2, 2, 4, 5, 8],
                 use_tanh: bool = True,
                 temperature: float = 1.0,
                 hifi_kernels: List[int] = [3, 7, 11, 15, 23, 32],
                 use_decoder_gaatn: bool = False,
                 freeze_encoder: bool = False,
                 # New params for creating encoder if not provided
                 code_dim: int = 128,
                 sample_rate: int = 24000,
                 n_res_blocks: int = 2,
                 n_conformer: int = 2,
                 conformer_heads: int = 8):
        super().__init__()
        
        # Create or use provided JEPA encoder
        if jepa_encoder is None:
            self.encoder = JEPAEncoder(
                sample_rate=sample_rate,
                code_dim=code_dim,
                channels=channels,
                strides=strides,
                n_res_blocks=n_res_blocks,
                n_conformer=n_conformer,
                conformer_heads=conformer_heads,
                use_gaatn=True,
                debug=True
            )
        else:
            self.encoder = jepa_encoder
        
        code_dim = self.encoder.code_dim
        self.sample_rate = self.encoder.sample_rate
        self.strides = strides
        self.hop_length = self.encoder.hop_length
        self.code_dim = code_dim
        
        # FSQ quantizer
        self.fsq = FiniteScalarQuantizer(
            levels=fsq_levels, dim=code_dim, normalized=True,
            use_tanh=use_tanh, temperature=temperature
        )
        
        # Decoder
        self.bottleneck_unproj = nn.Conv1d(code_dim, channels[-1], 1)
        self.decoder = nn.ModuleList([
            HiFiDecoderBlock(channels[i+1], channels[i], strides[i], hifi_kernels, use_decoder_gaatn)
            for i in range(len(strides)-1, -1, -1)
        ])
        
        self.output_conv = nn.Conv1d(channels[0], 1, kernel_size=7, padding=3)
        self.final_activation = nn.Tanh()
        
        self.fsq_levels = fsq_levels
        self._last_dec_attn_maps: List[torch.Tensor] = []
        
        # Only init decoder weights, not encoder
        for m in [self.bottleneck_unproj, self.output_conv, *self.decoder]:
            self._init_weights(m)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: 
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Module):
            for subm in m.modules():
                if isinstance(subm, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                    nn.init.trunc_normal_(subm.weight, std=0.02)
                    if subm.bias is not None:
                        nn.init.zeros_(subm.bias)
    
    def encode(self, wav: torch.Tensor):
        """Encode using JEPA encoder (frozen via lr=0 in optimizer)"""
        # Encoder will use lr=0, so gradients won't update it
        z_e = self.encoder.encode(wav)
        z_q, indices, aux_loss = self.fsq(z_e)
        return z_q, z_e, indices, aux_loss
    
    def decode(self, z_q: torch.Tensor):
        """Decode quantized representations to waveform"""
        x = self.bottleneck_unproj(z_q)
        self._last_dec_attn_maps = []
        
        for dec in self.decoder:
            x, gate = dec(x)
            if not self.training and gate is not None:
                self._last_dec_attn_maps.append(gate.detach())
        
        wav = self.output_conv(x)
        wav = self.final_activation(wav)
        return wav
    
    def forward(self, wav: torch.Tensor):
        original_length = wav.shape[-1]
        z_q, z_e, indices, aux_loss = self.encode(wav)
        rec = self.decode(z_q)
        
        if rec.shape[-1] > original_length:
            rec = rec[..., :original_length]
        elif rec.shape[-1] < original_length:
            rec = F.pad(rec, (0, original_length - rec.shape[-1]))
        
        return rec, indices, aux_loss, z_e
    
    def get_decoder_attention_maps(self):
        return self._last_dec_attn_maps

# ------------------------------
# Discriminators
# ------------------------------

class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (5,1), (3,1), padding=(2,0)),
            nn.Conv2d(32,128,(5,1),(3,1), padding=(2,0)),
            nn.Conv2d(128,512,(5,1),(3,1),padding=(2,0)),
            nn.Conv2d(512,1024,(5,1),(3,1),padding=(2,0)),
            nn.Conv2d(1024,1024,(5,1),1,padding=(2,0)),
        ])
        self.post = nn.Conv2d(1024,1,(3,1),1,padding=(1,0))
    
    def forward(self, x):
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0,n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t//self.period, self.period)
        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x),0.1)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1), fmaps

class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1,16,15,1,padding=7),
            nn.Conv1d(16,64,41,4,groups=4,padding=20),
            nn.Conv1d(64,256,41,4,groups=16,padding=20),
            nn.Conv1d(256,1024,41,4,groups=64,padding=20),
            nn.Conv1d(1024,1024,41,4,groups=256,padding=20),
            nn.Conv1d(1024,1024,5,1,padding=2),
        ])
        self.post = nn.Conv1d(1024,1,3,1,padding=1)
    
    def forward(self, x):
        fmaps = []
        for c in self.convs:
            x = F.leaky_relu(c(x),0.1)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1), fmaps

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=[2,3,5,7,11]):
        super().__init__()
        self.ds = nn.ModuleList([PeriodDiscriminator(p) for p in periods])
    
    def forward(self, y, y_hat):
        rs, gs, fr, fg = [],[],[],[]
        for d in self.ds:
            r, fr_ = d(y)
            g, fg_ = d(y_hat)
            rs.append(r)
            gs.append(g)
            fr.append(fr_)
            fg.append(fg_)
        return rs, gs, fr, fg

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.ds = nn.ModuleList([ScaleDiscriminator(), ScaleDiscriminator(), ScaleDiscriminator()])
        self.pools = nn.ModuleList([nn.Identity(), nn.AvgPool1d(4,2,padding=2), nn.AvgPool1d(4,2,padding=2)])
    
    def forward(self, y, y_hat):
        rs,gs,fr,fg = [],[],[],[]
        for i,(d,p) in enumerate(zip(self.ds,self.pools)):
            yy, gg = y, y_hat
            if i > 0:
                yy = p(y)
                gg = p(y_hat)
            r, fr_ = d(yy)
            g, fg_ = d(gg)
            rs.append(r)
            gs.append(g)
            fr.append(fr_)
            fg.append(fg_)
        return rs,gs,fr,fg

def make_collate_fn(sample_rate, hop_length):
    """Create a collate function that ensures minimum sequence length"""
    def collate_fn(batch):
        if not batch: 
            return None
        
        # Get maximum sequence length in batch
        T = max(x.shape[0] for x in batch)
        
        # Ensure minimum sequence length to prevent encoder dimension collapse
        # Rule: output_time = ceil(input_time / hop_length)
        # To get output_time >= 4, we need input_time >= 3*hop_length + 1
        min_samples = max(int(sample_rate * 0.5), 4 * hop_length)
        T = max(T, min_samples)
        
        # Align to hop_length boundary
        T = ((T + hop_length - 1) // hop_length) * hop_length
        
        # Stack and add channel dimension
        xs = torch.stack([F.pad(x, (0, T - x.shape[0])) for x in batch], dim=0)
        return xs.unsqueeze(1)  # [B, 1, T]
    
    return collate_fn

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr,dg in zip(fmap_r,fmap_g):
        for rl,gl in zip(dr,dg):
            loss += F.l1_loss(gl, rl.detach())
    return loss*2

def discriminator_loss(dr_list, dg_list):
    loss = 0
    for dr,dg in zip(dr_list,dg_list):
        loss += F.mse_loss(dr, torch.ones_like(dr))
        loss += F.mse_loss(dg, torch.zeros_like(dg))
    return loss

def generator_loss(dg_list):
    loss = 0
    for dg in dg_list:
        loss += F.mse_loss(dg, torch.ones_like(dg))
    return loss

# ------------------------------
# Training
# ------------------------------

def build_engine(model: nn.Module, optimizer, ds_config, args):
    if isinstance(ds_config, str):
        with open(ds_config, "r") as f:
            ds_config = json.load(f)
    if torch.cuda.is_available() and args.local_rank is not None and args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
    engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args, model=model, optimizer=optimizer,
        model_parameters=model.parameters(), config=ds_config
    )
    return engine, scheduler

def train_jepa_encoder(args):
    """Stage 1: PURE JEPA - No artificial regularization"""
    ensure_dir(args.out_dir)
    
    rank = dist.get_rank() if is_distributed() else 0
    world = dist.get_world_size() if is_distributed() else 1
    
    dataset = StreamingWaveformDataset(
        root_dir=args.jsonl, sample_rate=args.sample_rate,
        max_seconds=args.max_seconds, sleep=5.0,
        rank=rank, world_size=world, augment=True
    )
    
    channels = [int(x) for x in args.channels.split(',')]
    strides = [int(x) for x in args.strides.split(',')]
    hop = math.prod(strides)
    
    collate_fn = make_collate_fn(args.sample_rate, hop)
    
    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=0,
                    pin_memory=True, collate_fn=collate_fn)
    
    model = JEPAEncoder(
        sample_rate=args.sample_rate,
        code_dim=args.code_dim,
        channels=channels,
        strides=strides,
        n_res_blocks=args.n_res_blocks,
        n_conformer=args.n_conformer,
        conformer_heads=args.heads,
        use_gaatn=True
    )

    if rank0():
        print_model_stats(model, "JEPA Encoder (Stage 1)")
        #input("Press Enter to continue training...")  # Pause to read

    # FIX 1: Set lr=0 for target encoder params
    online_params = [
        p for n, p in model.named_parameters() 
        if not n.startswith('target_encoder.')
    ]
    
    target_params = [
        p for n, p in model.named_parameters() 
        if n.startswith('target_encoder.')
    ]
    
    opt = torch.optim.AdamW([
        {'params': online_params, 'lr': args.lr, 'weight_decay': 1e-3},
        {'params': target_params, 'lr': 0.0, 'weight_decay': 0.0}  # lr=0 for target
    ], betas=(0.8, 0.99))
    
    if rank0():
        params = sum(p.numel() for p in model.parameters())
        print(f"[JEPA Stage 1] Encoder Parameters: {params/1e6:.1f}M")
        print(f"[JEPA Stage 1] Hop length (total stride): {hop}")
        print(f"[JEPA Stage 1] Training PURE JEPA (target encoder lr=0)")
        print(f"Training for {args.max_steps} steps")
    
    engine, _ = build_engine(model, opt, args.ds_config, args)
    device = engine.device
    model_dtype = next(engine.module.parameters()).dtype
    
    # FIX 2: Resume functionality for stage 1 with backwards compatibility
    ckpt_dir = os.path.join(args.out_dir, "jepa_encoder_ds")
    global_step = 0
    
    if args.resume and os.path.isdir(ckpt_dir):
        try:
            load_path, client_sd = engine.load_checkpoint(ckpt_dir, tag=None)
            
            # Handle global_step from new or old checkpoints
            if client_sd is not None and 'global_step' in client_sd:
                global_step = client_sd['global_step']
                if rank0():
                    print(f"[JEPA] Resumed from {load_path} at step {global_step}")
            else:
                # Old checkpoint format - try to infer from checkpoint name
                if load_path and 'step' in str(load_path):
                    import re
                    match = re.search(r'step(\d+)', str(load_path))
                    if match:
                        global_step = int(match.group(1))
                        if rank0():
                            print(f"[JEPA] Old checkpoint format detected. Inferred step {global_step} from checkpoint name")
                    else:
                        global_step = 0
                        if rank0():
                            print(f"[JEPA] Old checkpoint format detected. Starting from step 0")
                else:
                    global_step = 0
                    if rank0():
                        print(f"[JEPA] Resumed from {load_path} (old format, starting from step 0)")
                        
        except RuntimeError as e:
            if "optimizer" in str(e).lower():
                # Optimizer state dict mismatch - try loading model only
                if rank0():
                    print(f"[JEPA] Optimizer format changed, attempting to load model weights only...")
                try:
                    # DeepSpeed may have different checkpoint structure
                    import glob
                    model_files = glob.glob(os.path.join(ckpt_dir, "**/pytorch_model.bin"), recursive=True)
                    if not model_files:
                        model_files = glob.glob(os.path.join(ckpt_dir, "**/mp_rank_*_model_states.pt"), recursive=True)
                    
                    if model_files:
                        # Load just the model state
                        state_dict = torch.load(model_files[0], map_location=device)
                        engine.module.load_state_dict(state_dict, strict=False)
                        global_step = 0
                        if rank0():
                            print(f"[JEPA] Loaded model weights only from old checkpoint, starting optimizer fresh")
                    else:
                        if rank0():
                            print(f"[JEPA] Could not find model weights, starting fresh")
                except Exception as load_error:
                    if rank0():
                        print(f"[JEPA] Failed to load old checkpoint: {load_error}, starting fresh")
            else:
                raise
        except Exception as e:
            if rank0():
                print(f"[JEPA] Error loading checkpoint: {e}, starting fresh")
    
    loss_w = deque(maxlen=10)
    pbar = tqdm(dl, total=args.max_steps - global_step, 
                desc=f"JEPA Encoder Training (from step {global_step})", 
                disable=not rank0())
    
    for step, wav in enumerate(pbar):
        if global_step >= args.max_steps: 
            break
        if wav is None: 
            continue
        
        try:
            wav = wav.to(device=device, dtype=model_dtype)
            B, C, T_wav = wav.shape

            # Get actual T_z from encoding
            with torch.no_grad():
                z_tmp = engine.module.encode(wav)
            T_z = z_tmp.shape[-1]

            # Create JEPA mask (1=visible, 0=masked)
            mask = create_jepa_mask(
                batch_size=B, seq_len=T_z, mask_ratio=args.mask_ratio,
                min_span=2, max_span=max(4, T_z // 4), device=device
            )
            
            # Forward pass
            z_context, z_pred, mask, z_target = engine.module(wav, mask)

            # ✅ PURE JEPA LOSS: Only MSE on masked regions
            mask_for_loss = (1 - mask).unsqueeze(1)  # [B, 1, T_z]

            # masked-only MSE with proper normalization
            w = (1 - mask).unsqueeze(1).to(z_pred.dtype)        # [B,1,T], 1 on masked, 0 on visible
            diff2 = (z_pred - z_target.detach())**2             # [B,C,T]
            num = (diff2 * w).sum()
            den = (w.sum() * z_pred.shape[1]).clamp_min(1e-8)   # masked tokens * channels
            loss = num / den

            
            total_loss = loss
            
            # ✅ Optional: Monitor collapse WITHOUT backprop
            with torch.no_grad():
                # Statistics for monitoring (not used in loss)
                std_per_dim = z_pred.std(dim=[0, 2])
                mean_std = std_per_dim.mean().item()
                min_std = std_per_dim.min().item()
                
                # Log warning if collapse detected
                if mean_std < 0.01 and rank0() and (global_step % 100) == 0:
                    print(f"[WARNING] Potential collapse detected: mean_std={mean_std:.6f}")
            
            loss_w.append(float(total_loss.detach().item()))
            
            engine.backward(total_loss)
            engine.step()

            # Update EMA target encoder
            engine.module.update_target_encoder()
            
            global_step += 1
            
            # Logging
            if rank0() and (global_step % args.log_every) == 0:
                avg_loss = sum(loss_w)/len(loss_w) if loss_w else 0
                num_masked = int((mask == 0).sum().item())
                
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    T_z=T_z,
                    masked=f"{int((mask == 0).sum().item())}/{B*T_z}",
                    std=f"{mean_std:.4f}"
                )
                
                with open(os.path.join(args.out_dir, 'jepa_logs.txt'), 'a') as f:
                    f.write(f"{global_step}\t{avg_loss:.6f}\t{mean_std:.6f}\t{min_std:.6f}\n")
            
            # Save checkpoint with global_step
            if args.save_every_steps > 0 and (global_step % args.save_every_steps == 0):
                client_sd = {'global_step': global_step}
                engine.save_checkpoint(ckpt_dir, tag=f"step{global_step}", client_state=client_sd)
                if rank0():
                    print(f"[JEPA CHECKPOINT] Saved at step {global_step}")
        
        except Exception as e:
            if rank0():
                print(f"\n[ERROR at step {global_step}] {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            raise
    
    # Final save
    client_sd = {'global_step': global_step}
    engine.save_checkpoint(ckpt_dir, tag="final", client_state=client_sd)
    if rank0():
        print(f"[JEPA] Training complete after {global_step} steps.")

def train_decoder_with_frozen_encoder(args):
    """Stage 2: Train FSQ + Decoder with frozen JEPA encoder"""
    ensure_dir(args.out_dir)
    
    mr_stft = MRSTFTLoss(
        fft_sizes=[2048, 1024, 512, 256, 128],
        hop_sizes=[512, 256, 128, 64, 32],
        win_lengths=[2048, 1024, 512, 256, 128]
    )
    
    fsq_levels = [int(x) for x in args.fsq_levels.split(',')]
    channels = [int(x) for x in args.channels.split(',')]
    strides = [int(x) for x in args.strides.split(',')]
    
    rank = dist.get_rank() if is_distributed() else 0
    world = dist.get_world_size() if is_distributed() else 1
    
    dataset = StreamingWaveformDataset(
        root_dir=args.jsonl, sample_rate=args.sample_rate,
        max_seconds=args.max_seconds, sleep=5.0,
        rank=rank, world_size=world, augment=True
    )
    
    hop = math.prod(strides)
    
    # Create the collate function with proper arguments
    collate_fn = make_collate_fn(args.sample_rate, hop)
    
    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=0,
                    pin_memory=True, collate_fn=collate_fn)
    
    # Load pretrained JEPA encoder
    jepa_encoder = JEPAEncoder(
        sample_rate=args.sample_rate,
        code_dim=args.code_dim,
        channels=channels,
        strides=strides,
        n_res_blocks=args.n_res_blocks,
        n_conformer=args.n_conformer,
        conformer_heads=args.heads,
        use_gaatn=True
    )
    
    # Load JEPA checkpoint
    jepa_ckpt_dir = os.path.join(args.out_dir, "jepa_encoder_ds")
    if not os.path.exists(jepa_ckpt_dir):
        raise RuntimeError(f"JEPA checkpoint not found at {jepa_ckpt_dir}. Run stage 1 first!")
    
    # Load JEPA weights without DeepSpeed to avoid parameter state issues
    jepa_encoder_state = None
    try:
        # Find the checkpoint files
        import glob
        checkpoint_files = glob.glob(os.path.join(jepa_ckpt_dir, "**/*.pt"), recursive=True)
        if not checkpoint_files:
            checkpoint_files = glob.glob(os.path.join(jepa_ckpt_dir, "**/*.bin"), recursive=True)
        
        if checkpoint_files:
            # Load the state dict directly
            for ckpt_file in checkpoint_files:
                if "model" in ckpt_file.lower() or "pytorch" in ckpt_file.lower():
                    jepa_encoder_state = torch.load(ckpt_file, map_location='cpu')
                    if rank0():
                        print(f"[Stage 2] Loaded JEPA encoder state from {ckpt_file}")
                    break
        
        if jepa_encoder_state is None:
            # Try loading through DeepSpeed as fallback
            online_params = [
                p for n, p in jepa_encoder.named_parameters() 
                if not n.startswith('target_encoder.')
            ]
            target_params = [
                p for n, p in jepa_encoder.named_parameters() 
                if n.startswith('target_encoder.')
            ]
            
            dummy_opt = torch.optim.AdamW([
                {'params': online_params, 'lr': 1e-8},
                {'params': target_params, 'lr': 0.0}
            ], betas=(0.8, 0.99))
            
            jepa_engine, _ = build_engine(jepa_encoder, dummy_opt, args.ds_config, args)
            jepa_engine.load_checkpoint(jepa_ckpt_dir, tag=None)
            
            # Extract state dict from the engine
            jepa_encoder_state = jepa_engine.module.state_dict()
            
            if rank0():
                print(f"[Stage 2] Loaded JEPA encoder via DeepSpeed")
    except Exception as e:
        raise RuntimeError(f"Failed to load JEPA encoder checkpoint: {e}")
    
    # Load the state dict into the encoder
    if jepa_encoder_state is not None:
        jepa_encoder.load_state_dict(jepa_encoder_state, strict=False)
        if rank0():
            print(f"[Stage 2] Successfully loaded JEPA encoder weights")
    
    # Create full model with frozen encoder
    model = WaveformJEPAFSQVAE(
        jepa_encoder=jepa_encoder,  # Use the encoder with loaded weights directly
        fsq_levels=fsq_levels,
        channels=channels,
        strides=strides,
        use_tanh=args.use_tanh,
        temperature=args.temperature,
        hifi_kernels=[3, 7, 11, 15, 23, 32],
        use_decoder_gaatn=False,
        freeze_encoder=False  # Don't freeze here - use lr=0 instead
    )

    if rank0():
        print_model_stats(model, "Full Model (Stage 2: FSQ + Decoder)")
        
        # Additional breakdown for stage 2
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        decoder_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith('encoder.'))
        
        print("\n" + "="*80)
        print("Training Configuration (Stage 2)")
        print("="*80)
        print(f"{'Component':<40} {'Parameters':<20} {'LR':<15}")
        print("-"*80)
        print(f"{'JEPA Encoder (frozen via lr=0)':<40} {encoder_params/1e6:>6.2f}M          0.0")
        print(f"{'FSQ + HiFi-GAN Decoder (trainable)':<40} {decoder_params/1e6:>6.2f}M          {args.lr}")
        print("="*80 + "\n")
        
        #input("Press Enter to continue training...")  # Pause to read
    
    # Ensure FSQ boundaries are on the correct device
    if torch.cuda.is_available():
        device_placeholder = torch.cuda.current_device()
        for boundary_module in model.fsq.boundaries:
            boundary_module.bounds = boundary_module.bounds.cuda(device_placeholder)
    
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    
    # FIX 1: Separate parameter groups: encoder gets lr=0, decoder gets normal lr
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    
    opt_g = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 0.0},  # Frozen via lr=0
        {'params': decoder_params, 'lr': args.lr, 'weight_decay': 1e-3}
    ], betas=(0.8, 0.99))
    
    opt_d = torch.optim.AdamW(list(mpd.parameters())+list(msd.parameters()),
                              lr=args.lr*0.5, weight_decay=1e-3, betas=(0.8,0.99))
    
    if rank0():
        total_params = sum(p.numel() for p in model.parameters())
        encoder_param_count = sum(p.numel() for p in encoder_params)
        decoder_param_count = sum(p.numel() for p in decoder_params)
        print(f"[Stage 2] Total Parameters: {total_params/1e6:.1f}M")
        print(f"[Stage 2] Trainable (Decoder+FSQ): {decoder_param_count/1e6:.1f}M")
        print(f"[Stage 2] Frozen via lr=0 (JEPA Encoder): {encoder_param_count/1e6:.1f}M")
        print(f"[Stage 2] Hop length: {hop}, Min input samples: {4 * hop}")
    
    engine_g, _ = build_engine(model, opt_g, args.ds_config, args)
    device = engine_g.device
    model_dtype = next(engine_g.module.parameters()).dtype
    
    mpd = mpd.to(device=device, dtype=model_dtype)
    msd = msd.to(device=device, dtype=model_dtype)
    mr_stft = mr_stft.to(device) if torch.cuda.is_available() else mr_stft
    
    # FIX 2: Resume functionality for stage 2 with backwards compatibility
    ckpt_dir = os.path.join(args.out_dir, "decoder_ds")
    global_step = 0
    
    if args.resume and os.path.isdir(ckpt_dir):
        try:
            load_path, client_sd = engine_g.load_checkpoint(ckpt_dir, tag=None)
            
            # Handle global_step from new or old checkpoints
            if client_sd is not None and 'global_step' in client_sd:
                global_step = client_sd['global_step']
            else:
                # Old checkpoint format - try to infer from checkpoint name
                if load_path and 'step' in str(load_path):
                    import re
                    match = re.search(r'step(\d+)', str(load_path))
                    if match:
                        global_step = int(match.group(1))
                        if rank0():
                            print(f"[Stage 2] Old checkpoint format detected. Inferred step {global_step} from checkpoint name")
                    else:
                        global_step = 0
                        if rank0():
                            print(f"[Stage 2] Old checkpoint format detected. Starting from step 0")
                else:
                    global_step = 0
                    if rank0():
                        print(f"[Stage 2] Resumed from {load_path} (old format, starting from step 0)")
            
            # Load discriminator states
            disc_ckpt_path = os.path.join(ckpt_dir, "discriminators.pt")
            if os.path.exists(disc_ckpt_path):
                try:
                    s = torch.load(disc_ckpt_path, map_location=device)
                    mpd.load_state_dict(s['mpd'])
                    msd.load_state_dict(s['msd'])
                    opt_d.load_state_dict(s['optimizer_d'])
                    if rank0():
                        print(f"[Stage 2] Loaded discriminator states")
                except RuntimeError as e:
                    if "optimizer" in str(e).lower():
                        # Optimizer format mismatch, just load discriminator models
                        mpd.load_state_dict(s['mpd'])
                        msd.load_state_dict(s['msd'])
                        if rank0():
                            print(f"[Stage 2] Loaded discriminator models only (optimizer format changed)")
            
            if rank0():
                print(f"[Stage 2] Resumed from {load_path} at step {global_step}")
                
        except RuntimeError as e:
            if "optimizer" in str(e).lower():
                # Optimizer state dict mismatch - try loading model only
                if rank0():
                    print(f"[Stage 2] Optimizer format changed, attempting to load model weights only...")
                try:
                    # Try to load just the model state
                    import glob
                    model_files = glob.glob(os.path.join(ckpt_dir, "**/pytorch_model.bin"), recursive=True)
                    if not model_files:
                        model_files = glob.glob(os.path.join(ckpt_dir, "**/mp_rank_*_model_states.pt"), recursive=True)
                    
                    if model_files:
                        state_dict = torch.load(model_files[0], map_location=device)
                        engine_g.module.load_state_dict(state_dict, strict=False)
                        global_step = 0
                        
                        # Also try to load discriminators without optimizer
                        disc_ckpt_path = os.path.join(ckpt_dir, "discriminators.pt")
                        if os.path.exists(disc_ckpt_path):
                            s = torch.load(disc_ckpt_path, map_location=device)
                            mpd.load_state_dict(s['mpd'])
                            msd.load_state_dict(s['msd'])
                        
                        if rank0():
                            print(f"[Stage 2] Loaded model weights only from old checkpoint, starting optimizers fresh")
                    else:
                        if rank0():
                            print(f"[Stage 2] Could not find model weights, loading JEPA encoder only")
                except Exception as load_error:
                    if rank0():
                        print(f"[Stage 2] Failed to load old checkpoint: {load_error}, loading JEPA encoder only")
            else:
                raise
        except Exception as e:
            if rank0():
                print(f"[Stage 2] Error loading checkpoint: {e}, loading JEPA encoder only")
    
    loss_w = deque(maxlen=10)
    rec_w = deque(maxlen=10)
    stft_w = deque(maxlen=10)
    gen_w = deque(maxlen=10)
    disc_w = deque(maxlen=10)
    
    samples_dir = ensure_dir(os.path.join(args.out_dir, "samples"))
    pbar = tqdm(dl, total=args.max_steps - global_step, 
                desc=f"Stage 2: Decoder Training (from step {global_step})", 
                disable=not rank0())
    
    for step, wav in enumerate(pbar):
        if global_step >= args.max_steps: 
            break
        if wav is None: 
            continue
        
        try:
            wav = wav.to(device=device, dtype=model_dtype)
            
            # Generator
            opt_g.zero_grad()
            rec, indices, aux_loss, z_e = engine_g(wav)
            
            rec_loss = F.l1_loss(rec, wav)
            stft_loss = mr_stft(rec, wav)
            
            y_df_r, y_df_g, f_df_r, f_df_g = mpd(wav, rec)
            y_ds_r, y_ds_g, f_ds_r, f_ds_g = msd(wav, rec)
            
            gen_loss_f = generator_loss(y_df_g)
            gen_loss_s = generator_loss(y_ds_g)
            feat_loss_f = feature_loss(f_df_r, f_df_g)
            feat_loss_s = feature_loss(f_ds_r, f_ds_g)
            gen_loss = gen_loss_f + gen_loss_s + feat_loss_f + feat_loss_s
            
            total_g = rec_loss + args.spectral_weight*stft_loss + args.gan_weight*gen_loss
            
            loss_w.append(float(total_g.detach().item()))
            rec_w.append(float(rec_loss.detach().item()))
            stft_w.append(float(stft_loss.detach().item()))
            gen_w.append(float(gen_loss.detach().item()))
            
            engine_g.backward(total_g)
            engine_g.step()
            
            # Discriminator
            if global_step > args.disc_start_step and (global_step % args.disc_interval) == 0:
                opt_d.zero_grad()
                with torch.no_grad():
                    rec_det = rec.detach()
                y_df_r, y_df_g, _, _ = mpd(wav, rec_det)
                y_ds_r, y_ds_g, _, _ = msd(wav, rec_det)
                dloss = discriminator_loss(y_df_r, y_df_g) + discriminator_loss(y_ds_r, y_ds_g)
                disc_w.append(float(dloss.detach().item()))
                dloss.backward()
                opt_d.step()
            
            global_step += 1
            
            if rank0() and (global_step % args.log_every) == 0:
                avg_loss = sum(loss_w)/len(loss_w) if loss_w else 0
                avg_rec = sum(rec_w)/len(rec_w) if rec_w else 0
                avg_stft = sum(stft_w)/len(stft_w) if stft_w else 0
                avg_gen = sum(gen_w)/len(gen_w) if gen_w else 0
                avg_disc = sum(disc_w)/len(disc_w) if disc_w else 0
                
                pbar.set_postfix(loss=f"{avg_loss:.4f}", rec=f"{avg_rec:.4f}",
                               stft=f"{avg_stft:.4f}", gen=f"{avg_gen:.4f}",
                               disc=f"{avg_disc:.4f}")
                
                with open(os.path.join(args.out_dir, 'decoder_logs.txt'), 'a') as f:
                    f.write(f"{global_step}\t{avg_loss:.6f}\t{avg_rec:.6f}\t{avg_stft:.6f}\t{avg_gen:.6f}\t{avg_disc:.6f}\n")
            
            # FIX 3: Inference in stage 2 with GatheredParameters
            if args.sample_every > 0 and args.sample_wav and (global_step % args.sample_every == 0):
                try:
                    import deepspeed
                    with torch.no_grad():
                        # Gather sharded params for ZeRO-3
                        with deepspeed.zero.GatheredParameters(list(engine_g.module.parameters()), modifier_rank=None):
                            sample_path = os.path.abspath(args.sample_wav)
                            test_wav, _ = load_mono_resample(sample_path, args.sample_rate)
                            test_wav = test_wav.unsqueeze(0).to(device=device, dtype=model_dtype)
                            pad = (engine_g.module.hop_length - test_wav.shape[-1] % engine_g.module.hop_length) % engine_g.module.hop_length
                            if pad > 0:
                                test_wav = F.pad(test_wav, (0, pad))
                            
                            # Capture indices for token stats
                            z_q, z_e, indices, _ = engine_g.module.encode(test_wav)
                            recv = engine_g.module.decode(z_q)
                            
                            # Compute FSQ token stats (G=7)
                            stats = fsq_token_stats_from_indices(
                                indices=indices,
                                fsq_levels=fsq_levels,
                                code_dim=engine_g.module.code_dim,
                                sample_rate=args.sample_rate,
                                strides=strides,
                                group_size=7
                            )
                    
                    if rank0():
                        # Write audio
                        out = recv.squeeze().float().cpu().numpy()
                        out = np.clip(out, -1, 1)
                        outp = os.path.join(samples_dir, f"step{global_step:07d}.wav")
                        soundfile.write(outp, out, args.sample_rate)
                        
                        print(f"[TOK] step={global_step} G={stats['G']} fps={stats['fps']:.4f} dur={stats['seconds']:.2f}s tps={stats['tokens_per_sec']:.3f}")
                        print(f"[SYNTH] {outp}")
                except Exception as e:
                    if rank0():
                        print(f"[SYNTH ERROR] {e}")
            
            # Save checkpoint with global_step
            if args.save_every_steps > 0 and (global_step % args.save_every_steps == 0):
                client_sd = {'global_step': global_step}
                engine_g.save_checkpoint(ckpt_dir, tag=f"step{global_step}", client_state=client_sd)
                torch.save({'mpd': mpd.state_dict(), 'msd': msd.state_dict(),
                           'optimizer_d': opt_d.state_dict()},
                          os.path.join(ckpt_dir, "discriminators.pt"))
                if rank0():
                    print(f"[CHECKPOINT] Saved at step {global_step}")
        
        except Exception as e:
            if rank0():
                print(f"\n[ERROR at step {global_step}] {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            raise
    
    # Final save
    client_sd = {'global_step': global_step}
    engine_g.save_checkpoint(ckpt_dir, tag="final", client_state=client_sd)
    if rank0():
        torch.save({'mpd': mpd.state_dict(), 'msd': msd.state_dict(),
                   'optimizer_d': opt_d.state_dict()},
                  os.path.join(ckpt_dir, "discriminators.pt"))
        print(f"[Stage 2] Training complete after {global_step} steps.")

# ------------------------------
# CLI
# ------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--stage', type=str, required=True,
                    choices=['train_jepa', 'train_decoder'])
    
    # Audio / Model
    ap.add_argument('--sample_rate', type=int, default=24000)
    ap.add_argument('--fsq_levels', type=str, default='4,4,4,4')
    ap.add_argument('--code_dim', type=int, default=128)
    ap.add_argument('--channels', type=str, default='64,128,256,384,512,512')
    ap.add_argument('--strides', type=str, default='8,8,5,5,6')
    ap.add_argument('--n_res_blocks', type=int, default=8)
    ap.add_argument('--n_conformer', type=int, default=8)
    ap.add_argument('--heads', type=int, default=16)
    ap.add_argument('--use_tanh', type=bool, default=True)
    ap.add_argument('--temperature', type=float, default=1.0)
    
    # JEPA specific
    ap.add_argument('--mask_ratio', type=float, default=0.5)
    
    # Train
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--max_steps', type=int, default=800000)
    ap.add_argument('--lr', type=float, default=1.5e-4)
    ap.add_argument('--max_seconds', type=float, default=15.0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--spectral_weight', type=float, default=2.0)
    ap.add_argument('--gan_weight', type=float, default=0.1)
    ap.add_argument('--disc_start_step', type=int, default=5000)
    ap.add_argument('--disc_interval', type=int, default=1)
    ap.add_argument('--sample_every', type=int, default=500)
    ap.add_argument('--sample_wav', type=str, default=None)
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--save_every_steps', type=int, default=1000)

    ap.add_argument('--reg_weight', type=float, default=1.0)

    
    # DeepSpeed
    ap.add_argument('--ds_config', type=str, required=True)
    ap.add_argument('--local_rank', type=int, default=-1)
    return ap.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    
    if args.stage == 'train_jepa':
        print("[STAGE 1] Training JEPA Encoder")
        train_jepa_encoder(args)
    elif args.stage == 'train_decoder':
        print("[STAGE 2] Training Decoder with Frozen JEPA Encoder")
        train_decoder_with_frozen_encoder(args)

if __name__ == '__main__':
    main()
