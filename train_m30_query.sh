#!/usr/bin/env bash
# Train with the object-level TrackQueryHead.
# Usage: bash train_m30_query.sh [epochs] [warmup_epochs]
set -e
cd "$(dirname "$0")"

EPOCHS="${1:-12}"
WARMUP="${2:-3}"
SEED="${SEED:-54}"
QUERIES="${QUERIES:-32}"
LOSS_WEIGHT="${LOSS_WEIGHT:-0.05}"
M26_CKPT="$PWD/checkpoints/m26_targetflow_m20e3_epoch_003_seed53.pt"
if [ "$QUERIES" = "32" ] && [ "$LOSS_WEIGHT" = "0.05" ]; then
  TRAIN_ROOT="$PWD/log/m30_query_m26e3_e${EPOCHS}_seed${SEED}"
else
  TRAIN_ROOT="$PWD/log/m30_query_q${QUERIES}_w${LOSS_WEIGHT}_e${EPOCHS}_seed${SEED}"
fi

python train_temporal_memory.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed="$SEED" TRAIN.epochs="$EPOCHS" TRAIN.batch_size=1 \
  TRAIN.lr=0.000001 TRAIN.scheduler=cosine TRAIN.scheduler_min_lr=0.0000001 \
  TRAIN.checkpoint_interval=1 TRAIN.model_save_root="$TRAIN_ROOT" \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_init_model_path="$M26_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_dense_sampling_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_dense_event_count_cutoff=200000 \
  TEMPORAL_MEMORY.temporal_memory_dense_view_multiplier=8 \
  TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_base_lr_multiplier=0.25 \
  TEMPORAL_MEMORY.temporal_memory_memory_lr_multiplier=0.25 \
  TEMPORAL_MEMORY.temporal_memory_attention_lr_multiplier=0.25 \
  TEMPORAL_MEMORY.temporal_memory_advection_alignment_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_advection_alignment_loss_weight=0.01 \
  TEMPORAL_MEMORY.temporal_memory_advection_alignment_lr_multiplier=4.0 \
  TEMPORAL_MEMORY.temporal_memory_advection_max_flow=2.0 \
  TEMPORAL_MEMORY.temporal_memory_advection_target_flow_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_advection_target_flow_weight=0.5 \
  TEMPORAL_MEMORY.temporal_memory_advection_target_flow_huber_delta=1.0 \
  TEMPORAL_MEMORY.temporal_memory_track_query_head_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_track_query_num_queries="$QUERIES" \
  TEMPORAL_MEMORY.temporal_memory_track_query_hidden=128 \
  TEMPORAL_MEMORY.temporal_memory_track_query_max_flow=2.0 \
  TEMPORAL_MEMORY.temporal_memory_track_query_loss_weight="$LOSS_WEIGHT" \
  TEMPORAL_MEMORY.temporal_memory_track_query_warmup_epochs="$WARMUP" \
  TEMPORAL_MEMORY.temporal_memory_track_query_lr_multiplier=1.0 \
  TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_enabled=false
