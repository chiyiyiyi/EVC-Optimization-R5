#!/usr/bin/env bash
# Evaluate any temporal-memory checkpoint with M26 + P41 + P32v7.
# Usage: bash eval_checkpoint_p32v7.sh /path/to/epoch_00X_seed54.pt
set -e
cd "$(dirname "$0")"

if [ -z "$1" ]; then
  echo "Usage: bash eval_checkpoint_p32v7.sh <checkpoint.pt>"
  exit 1
fi

M10_CKPT="$PWD/checkpoints/m10_dense_views2_epoch_002_seed42.pt"
M26_CKPT="$1"

python test2.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.eval=true TEST.roc=true TEST.prediction_threshold=0.7226 \
  TEMPORAL_FRAME.temporal_frame_enabled=false \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_model_path="$M26_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_secondary_model_path="$M10_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_secondary_max_event_count=30000 \
  TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0 \
  TEMPORAL_MEMORY.temporal_memory_inference_batch_size=8 \
  INFERENCE_TTA.p41_temporal_phase_enabled=true \
  INFERENCE_TTA.p41_temporal_phase_offset=25 \
  INFERENCE_TTA.p41_temporal_phase_original_weight=0.75 \
  INFERENCE_TTA.p41_temporal_phase_min_event_count=30000 \
  POSTPROCESS.p0_enabled=true POSTPROCESS.p0_spatial_radius=2 \
  POSTPROCESS.p0_temporal_bin_size=50 POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 POSTPROCESS.p0_min_duration_bins=5 \
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true \
  POSTPROCESS.p0c_retain_min_score=0.95 POSTPROCESS.p0b_enabled=false \
  POSTPROCESS.p18_score_track_recovery_enabled=true \
  POSTPROCESS.p18_event_count_cutoff=1 POSTPROCESS.p18_max_event_count=0 \
  POSTPROCESS.p18_candidate_floor=0.53 POSTPROCESS.p18_spatial_radius=5 \
  POSTPROCESS.p18_temporal_bin_size=50 POSTPROCESS.p18_max_link_distance=8.0 \
  POSTPROCESS.p18_max_gap_bins=1 POSTPROCESS.p18_min_track_bins=4 \
  POSTPROCESS.p18_restore_mode=best \
  POSTPROCESS.p6_density_threshold_enabled=true \
  POSTPROCESS.p6_event_count_cutoff=30000 \
  POSTPROCESS.p6_low_density_threshold=0.718 \
  POSTPROCESS.p6_high_density_threshold=0.7226 \
  POSTPROCESS.p32_track_quality_bonus_enabled=true \
  POSTPROCESS.p32_candidate_floor=0.60 \
  POSTPROCESS.p32_spatial_radius=2 \
  POSTPROCESS.p32_temporal_bin_size=50 \
  POSTPROCESS.p32_max_link_distance=8.0 \
  POSTPROCESS.p32_max_gap_bins=2 \
  POSTPROCESS.p32_min_track_bins=4 \
  POSTPROCESS.p32_min_seed_components=2 \
  POSTPROCESS.p32_bonus=0.010 \
  POSTPROCESS.p32_max_score_cap=0.97 \
  POSTPROCESS.p32_max_motion_residual=2.0 \
  POSTPROCESS.p32_velocity_history_bins=2
