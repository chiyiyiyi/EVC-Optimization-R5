"""Training loss for TrackQueryHead using target-ID track supervision."""

import numpy as np
import torch
import torch.nn.functional as F


def _fit_linear_velocity(bin_values, centroids):
    bin_values = np.asarray(bin_values, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    if bin_values.shape[0] < 2:
        return None
    design = np.column_stack((np.ones_like(bin_values), bin_values))
    x_coeff, _, _, _ = np.linalg.lstsq(design, centroids[:, 0], rcond=None)
    y_coeff, _, _, _ = np.linalg.lstsq(design, centroids[:, 1], rcond=None)
    return np.stack((x_coeff, y_coeff), axis=1)


def track_query_loss(
    predictions,
    anchors,
    labels,
    target_ids,
    event_time_indices,
    event_x,
    event_y,
    height,
    width,
    num_queries,
    sequence_length,
    objectness_weight=1.0,
    center_weight=1.0,
    velocity_weight=0.1,
):
    """Supervise objectness, centers and velocity with greedy anchor matching.

    ``anchors`` are normalized [K, 2] query anchors.  Each ground-truth target
    is matched to its nearest available anchor by average center distance.
    """
    objectness_logits = predictions["objectness"]  # [B, K]
    centers = predictions["centers"]  # [B, K, T, 2]
    velocity = predictions["velocity"]  # [B, K, 2]

    labels_np = labels.detach().cpu().numpy().reshape(-1)
    target_ids_np = target_ids.detach().cpu().numpy().reshape(-1).astype(np.int64)
    time_np = event_time_indices.detach().cpu().numpy().reshape(-1).astype(np.int64)
    x_np = event_x.detach().cpu().numpy().reshape(-1).astype(np.float64)
    y_np = event_y.detach().cpu().numpy().reshape(-1).astype(np.float64)
    anchors_np = anchors.detach().cpu().numpy()

    positive_indices = np.flatnonzero(labels_np >= 0.5)
    groups = {}
    for event_index in positive_indices:
        target_id = int(target_ids_np[event_index])
        if target_id <= 0:
            continue
        groups.setdefault(target_id, []).append(int(event_index))

    target_tracks = []
    for target_id, indices in groups.items():
        indices = np.asarray(indices, dtype=np.int64)
        bins = time_np[indices]
        unique_bins = np.unique(bins)
        if unique_bins.size < 2:
            continue
        positions = np.searchsorted(unique_bins, bins)
        counts = np.bincount(positions, minlength=unique_bins.size)
        centroids = np.column_stack(
            (
                np.bincount(
                    positions,
                    weights=x_np[indices] / max(width - 1, 1),
                    minlength=unique_bins.size,
                ),
                np.bincount(
                    positions,
                    weights=y_np[indices] / max(height - 1, 1),
                    minlength=unique_bins.size,
                ),
            )
        ) / counts[:, None]
        coeff = _fit_linear_velocity(unique_bins, centroids)
        if coeff is None:
            continue
        target_tracks.append((unique_bins, centroids, coeff))

    batch_size = objectness_logits.shape[0]
    device = objectness_logits.device
    objectness_target = torch.zeros_like(objectness_logits)
    center_target = torch.zeros_like(centers)
    velocity_target = torch.zeros_like(velocity)
    matched = 0

    used_queries = set()
    for bins, centroids, coeff in target_tracks:
        mean_center = centroids.mean(axis=0)
        distances = np.linalg.norm(anchors_np - mean_center, axis=1)
        order = np.argsort(distances)
        query_index = None
        for candidate in order:
            if int(candidate) not in used_queries:
                query_index = int(candidate)
                break
        if query_index is None:
            continue
        used_queries.add(query_index)
        matched += 1

        min_bin = int(bins.min())
        max_bin = int(bins.max())
        for batch_index in range(batch_size):
            objectness_target[batch_index, query_index] = 1.0
            for time_index in range(sequence_length):
                if not (min_bin <= time_index <= max_bin):
                    continue
                expected = np.asarray([1.0, float(time_index)]) @ coeff
                center_target[batch_index, query_index, time_index, 0] = expected[0]
                center_target[batch_index, query_index, time_index, 1] = expected[1]
            velocity_target[batch_index, query_index, 0] = coeff[1, 0]
            velocity_target[batch_index, query_index, 1] = coeff[1, 1]

    positive_weight = torch.where(
        objectness_target > 0.5,
        torch.full_like(objectness_target, 1.0),
        torch.full_like(objectness_target, 0.1),
    )
    objectness_loss = F.binary_cross_entropy_with_logits(
        objectness_logits,
        objectness_target,
        weight=positive_weight,
    )

    center_mask = (objectness_target > 0.5).unsqueeze(-1).unsqueeze(-1)
    if center_mask.any():
        center_loss = F.smooth_l1_loss(
            centers[center_mask.expand_as(centers)],
            center_target[center_mask.expand_as(center_target)],
        )
        velocity_loss = F.smooth_l1_loss(
            velocity[objectness_target > 0.5],
            velocity_target[objectness_target > 0.5],
        )
    else:
        center_loss = centers.sum() * 0.0
        velocity_loss = velocity.sum() * 0.0

    loss = (
        objectness_weight * objectness_loss
        + center_weight * center_loss
        + velocity_weight * velocity_loss
    )
    diagnostics = {
        "matched_tracks": matched,
        "objectness_loss": float(objectness_loss.detach().item()),
        "center_loss": float(center_loss.detach().item()),
        "velocity_loss": float(velocity_loss.detach().item()),
    }
    return loss, diagnostics
