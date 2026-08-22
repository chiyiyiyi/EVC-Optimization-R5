#!/usr/bin/env bash
# Generate submission for adopted TrackQueryHead 48-query e6 seed55 + P32v7.
# Prerequisite: source env_round4.sh, then export CUDA_VISIBLE_DEVICES=0.
set -e
cd "$(dirname "$0")"

OUTPUT_DIR="$PWD/log/challenge2/query48_e6_seed55_final"
mkdir -p "$OUTPUT_DIR"

M10_CKPT="$PWD/checkpoints/m10_dense_views2_epoch_002_seed42.pt"
M26_CKPT="$PWD/log/m30_query_q48_w0.05_e12_seed55/runs/20260822-171003_seed55_pid3442713/epoch_006_seed55.pt"

python submit_challenge2.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.challenge_output_dir="$OUTPUT_DIR" TEST.prediction_threshold=0.7226 \
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

cd "$OUTPUT_DIR"
zip -j ../query48_e6_seed55_final.zip val_*.txt
unzip -l ../query48_e6_seed55_final.zip
