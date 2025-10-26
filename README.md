# JEPA Audio Training & Checkpoint Converter

Complete toolkit for 2-stage JEPA (Joint-Embedding Predictive Architecture) audio training with FSQ-VAE and HiFi-GAN, plus DeepSpeed checkpoint conversion utilities.

## Overview

This repository contains two main components:

1. **`train_fsqvae_jepa.py`** - Two-stage self-supervised audio encoder training with neural vocoding
2. **`ds_ckpt_to_pt.py`** - DeepSpeed checkpoint consolidation utility

---

## Table of Contents

- [Installation](#installation)
- [Training Pipeline](#training-pipeline)
  - [Architecture Overview](#architecture-overview)
  - [Stage 1: JEPA Encoder Training](#stage-1-jepa-encoder-training)
  - [Stage 2: Decoder Training](#stage-2-decoder-training)
  - [Training Data Format](#training-data-format)
- [Checkpoint Conversion](#checkpoint-conversion)
- [Configuration](#configuration)
- [Common Workflows](#common-workflows)

---

## Installation

### Requirements

```bash
# Core dependencies
pip install torch torchaudio deepspeed --break-system-packages

# Audio processing
pip install soundfile librosa

# Utilities
pip install numpy tqdm
```

### System Requirements

- Python 3.8+
- CUDA-capable GPU(s)
- 16GB+ RAM per GPU recommended
- Multi-GPU setup recommended for distributed training

---

## Training Pipeline

### Architecture Overview

The training pipeline uses a 2-stage approach:

**Stage 1: Self-Supervised Encoder (JEPA)**
- Context encoder processes full audio sequences
- Predictor learns to reconstruct masked regions
- Self-supervised learning without labels
- Learns semantic audio representations

**Stage 2: Decoder with Frozen Encoder**
- JEPA encoder weights frozen
- FSQ-VAE quantizer + HiFi-GAN decoder trained
- Multi-scale discriminators (MPD + MSD)
- Spectral and adversarial losses

**Key Components:**
- **Gaussian Adaptive Attention (GAATN)**: Custom attention mechanism with learnable Gaussian weighting
- **FSQ Quantization**: Finite Scalar Quantization for discrete latent codes
- **Conformer Blocks**: Convolution-augmented transformer layers
- **HiFi-GAN**: High-fidelity neural vocoder

### Stage 1: JEPA Encoder Training

Train the self-supervised encoder using masked prediction:

```bash
CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port=29010 train_jepa_fsqvae_hifigan.py \
  --jsonl /path/to/audio_files.jsonl \
  --out_dir ./jepa_outputs \
  --stage train_jepa \
  --sample_rate 24000 \
  --batch_size 32 \
  --ds_config ds_config.json \
  --max_steps 200000 \
  --save_every_steps 1000 \
  --lr 1.5e-4 \
  --mask_ratio 0.5 \
  --resume
```

**Key Parameters:**
- `--mask_ratio`: Proportion of sequence to mask (default: 0.5)
- `--max_steps`: Total training steps for encoder
- `--batch_size`: Samples per GPU

**Output:**
- Checkpoints saved to `{out_dir}/jepa_encoder_ds/`
- Training logs in `{out_dir}/encoder_logs.txt`

### Stage 2: Decoder Training

Train the decoder with frozen encoder:

```bash
CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port=29010 train_jepa_fsqvae_hifigan.py \
  --jsonl /path/to/audio_files.jsonl \
  --out_dir ./jepa_outputs \
  --stage train_decoder \
  --sample_rate 24000 \
  --batch_size 8 \
  --ds_config ds_config.json \
  --sample_wav /path/to/test_audio.wav \
  --disc_start_step 5000 \
  --max_steps 800000 \
  --save_every_steps 1000 \
  --sample_every 500 \
  --spectral_weight 2.0 \
  --gan_weight 0.1 \
  --lr 1.5e-4
```

**Key Parameters:**
- `--disc_start_step`: Step to begin adversarial training
- `--sample_wav`: Audio file for periodic synthesis testing
- `--sample_every`: Generate samples every N steps
- `--spectral_weight`: Weight for multi-scale STFT loss
- `--gan_weight`: Weight for adversarial loss

**Output:**
- Generator checkpoints: `{out_dir}/decoder_ds/`
- Discriminator checkpoints: `{out_dir}/decoder_ds/discriminators.pt`
- Audio samples: `{out_dir}/samples/step0000XXX.wav`
- Training logs: `{out_dir}/decoder_logs.txt`

### Training Data Format

#### JSONL Structure

The training script expects a JSONL file where each line contains metadata for one audio file:

```jsonl
{"path": "/path/to/audio1.wav", "duration": 5.23, "speaker": "spk001"}
{"path": "/path/to/audio2.wav", "duration": 3.87, "speaker": "spk002"}
{"path": "/path/to/audio3.wav", "duration": 7.15, "speaker": "spk001"}
```

**Required Fields:**
- `path` (string): Absolute or relative path to audio file

**Optional Fields:**
- `duration` (float): Audio duration in seconds
- `speaker` (string): Speaker ID for multi-speaker datasets
- Any other metadata for filtering/analysis

**Format Requirements:**
- One JSON object per line
- Valid JSON syntax on each line
- UTF-8 encoding
- Unix-style newlines (`\n`)
- No commas between lines

#### Audio File Requirements

- **Format**: WAV, FLAC, MP3, or any format supported by `torchaudio`
- **Sample Rate**: Will be resampled to `--sample_rate` (default: 24000 Hz)
- **Channels**: Automatically converted to mono
- **Duration**: Clipped or padded to `--max_seconds` (default: 15.0s)

#### Example Dataset Preparation

```python
import json
import glob

# Generate JSONL from audio directory
audio_files = glob.glob("/data/audio/**/*.wav", recursive=True)

with open("train_data.jsonl", "w") as f:
    for audio_path in audio_files:
        entry = {"path": audio_path}
        f.write(json.dumps(entry) + "\n")
```

---

## Checkpoint Conversion

### Overview

`ds_ckpt_to_pt.py` converts DeepSpeed ZeRO-sharded checkpoints into single consolidated PyTorch `.pt` files. No DeepSpeed initialization required.

### Basic Usage

```bash
# Convert full checkpoint
python ds_ckpt_to_pt.py \
    --ds_dir ./jepa_outputs/jepa_encoder_ds \
    --out_pt ./encoder.pt

# Extract encoder only (for stage 2 checkpoints)
python ds_ckpt_to_pt.py \
    --ds_dir ./jepa_outputs/decoder_ds \
    --out_pt ./encoder_only.pt \
    --encoder_only

# Load specific checkpoint step
python ds_ckpt_to_pt.py \
    --ds_dir ./jepa_outputs/decoder_ds \
    --out_pt ./model_step50000.pt \
    --tag step50000
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--ds_dir` | Yes | DeepSpeed checkpoint directory |
| `--out_pt` | Yes | Output `.pt` file path |
| `--tag` | No | Checkpoint tag (`step50000`, `final`). Default: latest |
| `--encoder_only` | No | Extract only encoder weights |
| `--assert_gaatn` | No | Validate GAATN parameters present |

### Expected Checkpoint Structure

```
jepa_encoder_ds/
├── latest                          # Tag of latest checkpoint
├── step100000/                     # Checkpoint at step 100k
│   ├── mp_rank_00_model_states.pt  # Sharded model weights
│   ├── mp_rank_01_model_states.pt
│   └── zero_pp_rank_*_optim_states.pt
└── final/                          # Final checkpoint
    └── ...
```

---

## Configuration

### DeepSpeed Config (`ds_config.json`)

Example configuration for ZeRO Stage 2:

```json
{
  "train_batch_size": 64,
  "gradient_accumulation_steps": 2,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1.5e-4,
      "betas": [0.8, 0.99],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1.5e-4,
      "warmup_num_steps": 1000,
      "total_num_steps": 200000
    }
  }
}
```

### Model Hyperparameters

```bash
# Audio settings
--sample_rate 24000           # Target sample rate
--max_seconds 15.0           # Max audio duration

# Architecture
--code_dim 128               # FSQ latent dimension
--fsq_levels 4,4,4,4        # FSQ quantization levels
--channels 64,128,256,384,512,512  # Encoder channels
--strides 8,8,5,5,6         # Downsampling strides
--n_res_blocks 8            # Residual blocks per stage
--n_conformer 8             # Conformer layers
--heads 16                  # Attention heads

# Training
--batch_size 32             # Per-GPU batch size
--lr 1.5e-4                 # Learning rate
--max_steps 200000          # Total training steps
```

---

## Common Workflows

### Complete 2-Stage Training

```bash
# Stage 1: Train JEPA encoder (200k steps, ~2-3 days on 2x A100)
deepspeed train_jepa_fsqvae_hifigan.py \
  --jsonl train.jsonl \
  --out_dir ./outputs \
  --stage train_jepa \
  --batch_size 32 \
  --max_steps 200000 \
  --ds_config ds_config.json

# Convert encoder checkpoint
python ds_ckpt_to_pt.py \
  --ds_dir ./outputs/jepa_encoder_ds \
  --out_pt ./encoder_pretrained.pt \
  --tag final

# Stage 2: Train decoder (800k steps, ~5-7 days on 2x A100)
deepspeed train_jepa_fsqvae_hifigan.py \
  --jsonl train.jsonl \
  --out_dir ./outputs \
  --stage train_decoder \
  --batch_size 8 \
  --max_steps 800000 \
  --sample_wav test.wav \
  --sample_every 500 \
  --ds_config ds_config.json

# Convert final model
python ds_ckpt_to_pt.py \
  --ds_dir ./outputs/decoder_ds \
  --out_pt ./model_final.pt \
  --tag final
```

### Resume Training

```bash
# Resume from latest checkpoint
deepspeed train_jepa_fsqvae_hifigan.py \
  --jsonl train.jsonl \
  --out_dir ./outputs \
  --stage train_jepa \
  --resume \
  --ds_config ds_config.json
```

### Monitor Training

```bash
# Watch encoder logs
tail -f ./outputs/encoder_logs.txt

# Watch decoder logs
tail -f ./outputs/decoder_logs.txt

# Listen to synthesized samples
ls -lt ./outputs/samples/
```

### Token Statistics

During Stage 2 training, the script logs FSQ token statistics:

```
[TOK] step=5000 G=7 fps=93.75 dur=3.45s tps=656.250
```

- **G**: Number of token groups (7 per timestep)
- **fps**: Frames per second (temporal resolution)
- **dur**: Audio duration in seconds
- **tps**: Tokens per second

---

## Architecture Details

### FSQ Quantization

- **Levels**: `[4,4,4,4]` → 256 discrete codes per dimension
- **Code Dim**: 128 → 128-dimensional latent space
- **Group Size**: 7 → tokens packed in groups of 7
- **No codebook**: Deterministic quantization, no commitment loss

### Encoder Architecture

```
Input Audio (24kHz) 
  → Conv1D blocks with stride downsampling
  → Conformer layers (conv + attention)
  → Gaussian Adaptive Attention (GAATN)
  → Latent representation (128D)
```

### Decoder Architecture

```
Latent codes (128D)
  → FSQ quantization
  → Transposed Conv1D blocks with stride upsampling
  → Residual blocks
  → Output Audio (24kHz)
```

### Loss Functions

**Stage 1 (JEPA):**
- Masked region reconstruction loss (L1)

**Stage 2 (Decoder):**
- Reconstruction loss (L1 + L2)
- Multi-scale STFT loss (spectral convergence + magnitude)
- Adversarial loss (generator + discriminator)
- Feature matching loss

---

## Troubleshooting

### OOM Errors

- Reduce `--batch_size`
- Enable ZeRO Stage 3 in DeepSpeed config
- Reduce `--max_seconds` to process shorter clips

### Poor Audio Quality

- Increase `--spectral_weight` (try 5.0-10.0)
- Train longer (800k+ steps for stage 2)
- Delay discriminator start (`--disc_start_step 10000`)

### Checkpoint Loading Errors

```bash
# Verify checkpoint structure
ls -R ./outputs/jepa_encoder_ds/

# Try explicit tag
python ds_ckpt_to_pt.py --ds_dir ... --tag step50000
```

---

## Citation

If you use this code for audio self-supervised learning, please consider citing the JEPA paper:

```bibtex
@misc{ioannides2024densityadaptiveattentionneed,
      title={Density Adaptive Attention is All You Need: Robust Parameter-Efficient Fine-Tuning Across Multiple Modalities}, 
      author={Georgios Ioannides and Aman Chadha and Aaron Elkins},
      year={2024},
      eprint={2401.11143},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2401.11143}, 
}
```

**Last Updated**: October 2025
