JEPA 2-Stage Training Examples:

Stage 1 - Train JEPA Encoder:
CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port=29010 train_fsqvae_jepa2.py \
  --jsonl path \
  --out_dir ./jepa_outputs \
  --stage train_jepa \
  --sample_rate 24000 \
  --batch_size 32 \
  --ds_config ds_config.json \
  --max_steps 200000 \
  --save_every_steps 1000 \
  --lr 1.5e-4

Stage 2 - Train Decoder with Frozen Encoder:
CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port=29010 train_fsqvae_jepa2.py \
  --jsonl path \
  --out_dir ./jepa_outputs_v2 \
  --stage train_decoder \
  --sample_rate 24000 \
  --batch_size 8 \
  --ds_config ds_config.json \
  --sample_wav /home/gioannides/speechlm/original_audios/audio_00000062_018.wav \
  --disc_start_step 5000 \
  --max_steps 800000 \
  --save_every_steps 1000 \
  --lr 1.5e-4

Key Architecture Changes:
1. Stage 1: JEPA encoder learns representations via masked prediction
   - Context encoder processes full audio
   - Predictor predicts masked regions from visible context
   - Self-supervised learning without needing labels
   
2. Stage 2: Frozen encoder + trainable decoder
   - JEPA encoder weights are frozen
   - Only FSQ quantizer + HiFi-GAN decoder are trained
   - Uses discriminators and spectral losses for high-quality reconstruction

Benefits:
- Better representation learning through self-supervision
- Encoder learns semantic audio features before reconstruction
- Reduced training time for decoder (encoder already trained)
- Can pretrain encoder on larger unlabeled datasets
