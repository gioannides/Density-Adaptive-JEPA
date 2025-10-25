#!/usr/bin/env python
# ds_ckpt_to_pt.py
# Convert a DeepSpeed (ZeRO) checkpoint directory into a single .pt state_dict
# No DeepSpeed initialization, no MPI, no torch.distributed.

import os, argparse, glob, torch, json

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

def load_from_zero(checkpoint_dir: str, tag: str | None):
    """Load consolidated FP32 state_dict from a ZeRO shard directory."""
    sd = get_fp32_state_dict_from_zero_checkpoint(
        checkpoint_dir=checkpoint_dir,
        tag=tag   # e.g., 'final' or 'step100000'; None => latest
    )
    if not isinstance(sd, dict) or len(sd) == 0:
        raise RuntimeError(f"Empty state_dict from {checkpoint_dir} (tag={tag})")
    return sd

def maybe_load_from_consolidated_files(checkpoint_dir: str):
    """Fallback when a pt/bin exists (non-ZeRO or already merged)."""
    cand = []
    cand += glob.glob(os.path.join(checkpoint_dir, "**/pytorch_model.bin"), recursive=True)
    cand += glob.glob(os.path.join(checkpoint_dir, "**/*.pt"), recursive=True)
    if not cand:
        return None
    # Pick the largest file as best bet
    cand.sort(key=lambda p: os.path.getsize(p), reverse=True)
    try:
        return torch.load(cand[0], map_location="cpu")
    except Exception:
        return None

def filter_encoder_only(sd: dict) -> dict:
    """
    Return only encoder weights.
    Works for both Stage-2 (keys start with 'encoder.') and Stage-1 (pure encoder).
    """
    has_encoder_prefix = any(k.startswith("encoder.") for k in sd.keys())
    if has_encoder_prefix:
        return {k[len("encoder."):]: v for k,v in sd.items() if k.startswith("encoder.")}
    else:
        # Assume it’s already the encoder (Stage-1)
        return sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds_dir', required=True, help='DeepSpeed checkpoint dir (e.g. ./jepa_encoder_ds or ./decoder_ds)')
    ap.add_argument('--out_pt', required=True, help='Output .pt path')
    ap.add_argument('--tag', type=str, default=None, help="Checkpoint tag (e.g., 'final', 'step100000'); None => latest")
    ap.add_argument('--encoder_only', action='store_true', help='Save only the encoder submodule')
    ap.add_argument('--assert_gaatn', action='store_true', help='Fail if GAATN params are missing')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_pt)), exist_ok=True)

    # 1) Try ZeRO merge path (no DS init)
    try:
        sd = load_from_zero(args.ds_dir, args.tag)
        print(f"[info] loaded FP32 state_dict from ZeRO shards: {len(sd)} keys")
    except Exception as e:
        print(f"[warn] zero_to_fp32 merge failed: {e}")
        # 2) Fallback to any consolidated file we can find
        sd = maybe_load_from_consolidated_files(args.ds_dir)
        if sd is None:
            raise RuntimeError(f"Could not read any state_dict from {args.ds_dir}")

        # Some DS “consolidated” dumps wrap under extra nesting; unwrap if needed
        # Accept { 'module': state_dict } or { 'model_state_dict': state_dict } patterns
        if 'module' in sd and isinstance(sd['module'], dict):
            sd = sd['module']
        elif 'model_state_dict' in sd and isinstance(sd['model_state_dict'], dict):
            sd = sd['model_state_dict']
        print(f"[info] loaded fallback state_dict: {len(sd)} keys")

    # Optional: encoder-only
    if args.encoder_only:
        before = len(sd)
        sd = filter_encoder_only(sd)
        print(f"[info] encoder_only: {before} -> {len(sd)} keys")

    # Optional: sanity check GAATN params (so you don't accidentally drop them)
    if args.assert_gaatn:
        any_gaatn = any(("gaatn" in k.lower()) or ("gaussianadaptiveattention" in k.lower()) for k in sd.keys())
        if not any_gaatn:
            raise RuntimeError("assert_gaatn: GAATN-related parameters not found in state_dict")

    torch.save(sd, args.out_pt)
    print(f"[OK] wrote {args.out_pt}")

if __name__ == '__main__':
    main()

