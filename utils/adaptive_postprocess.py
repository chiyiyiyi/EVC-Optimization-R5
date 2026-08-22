"""Label-free, generalizable post-processing extensions for Challenge 2.

These modules are intentionally separate from ``postprocess.py``: they can be
stacked on top of the existing P0/P0c/P18 chain without changing the published
baseline.  They only consume event coordinates and continuous prediction
scores; they never read labels, target IDs, or video names.

P30 - DensityAdaptiveComponentFilter
------------------------------------
The teammate's 0.97 solution used a supervised RandomForest to delete pure
false-positive components.  That strategy is not valid for a hidden test set.
This module generalizes the same idea with conservative, label-free component
statistics:

* local density contrast against the video-wide event density,
* temporal fill ratio of a component,
* motion regularity (linear-fit residual of per-bin centroids).

Only components that violate several independent priors are considered
risky, and a global suppression cap prevents aggressive deletion.

P31 - MotionAwareTrackRecovery
------------------------------
High-speed, small, low-event targets often drop below the decision threshold
for one or two temporal bins.  P18 recovers weak components only when they are
close to a seed track in adjacent bins.  P31 fits a constant-velocity model to
seed-supported tracks and extrapolates their centroid across larger gaps,
then restores the best weak event on the predicted trajectory.
"""

from dataclasses import dataclass

import numpy as np


def _as_bool(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {'true', '1', 'yes', 'on'}:
            return True
        if value in {'false', '0', 'no', 'off'}:
            return False
        raise ValueError('Expected a boolean value, got {!r}.'.format(value))
    return bool(value)


def _spatial_components(coordinates, event_indices, spatial_radius):
    """Group 2D event coordinates into connected components in one bin."""
    unique_cells, inverse = np.unique(
        coordinates[:, :2],
        axis=0,
        return_inverse=True,
    )
    cell_lookup = {
        (int(cell[0]), int(cell[1])): index
        for index, cell in enumerate(unique_cells)
    }
    cell_events = [[] for _ in range(len(unique_cells))]
    for event_index, cell_index in enumerate(inverse):
        cell_events[int(cell_index)].append(event_index)

    neighbor_offsets = tuple(
        (dx, dy)
        for dx in range(-spatial_radius, spatial_radius + 1)
        for dy in range(-spatial_radius, spatial_radius + 1)
        if (dx, dy) != (0, 0)
    )
    visited = np.zeros(len(unique_cells), dtype=bool)
    components = []

    for start_index in range(len(unique_cells)):
        if visited[start_index]:
            continue
        stack = [start_index]
        visited[start_index] = True
        component_cells = []
        component_event_indices = []

        while stack:
            cell_index = stack.pop()
            component_cells.append(cell_index)
            component_event_indices.extend(cell_events[cell_index])
            x, y = unique_cells[cell_index]
            for dx, dy in neighbor_offsets:
                neighbor_index = cell_lookup.get((int(x + dx), int(y + dy)))
                if neighbor_index is not None and not visited[neighbor_index]:
                    visited[neighbor_index] = True
                    stack.append(neighbor_index)

        component_event_indices = np.asarray(
            component_event_indices,
            dtype=np.int64,
        )
        component_coordinates = coordinates[component_event_indices]
        components.append(
            {
                'event_indices': event_indices[component_event_indices],
                'cells': np.asarray(component_cells, dtype=np.int64),
                'centroid': component_coordinates[:, :2].mean(axis=0),
                'event_count': int(component_event_indices.size),
            }
        )
    return components


@dataclass(frozen=True)
class P30DensityComponentFilterConfig:
    enabled: bool = False
    spatial_radius: int = 2
    temporal_bin_size: int = 50
    temporal_radius_bins: int = 1
    min_component_events: int = 3
    min_component_duration_bins: int = 1
    preserve_high_confidence_score: float = 0.95
    local_density_contrast_enabled: bool = True
    local_density_contrast_min_ratio: float = 0.50
    motion_regularity_enabled: bool = True
    motion_regularity_min_bins: int = 4
    motion_regularity_max_residual: float = 3.0
    min_risk_criteria: int = 2
    max_suppression_fraction: float = 0.20

    def __post_init__(self):
        if self.spatial_radius < 0:
            raise ValueError('p30_spatial_radius must be non-negative.')
        if self.temporal_bin_size <= 0:
            raise ValueError('p30_temporal_bin_size must be positive.')
        if self.temporal_radius_bins < 0:
            raise ValueError('p30_temporal_radius_bins must be non-negative.')
        if self.min_component_events < 1:
            raise ValueError('p30_min_component_events must be positive.')
        if self.min_component_duration_bins < 1:
            raise ValueError(
                'p30_min_component_duration_bins must be positive.'
            )
        if not 0.0 <= self.preserve_high_confidence_score <= 1.0:
            raise ValueError(
                'p30_preserve_high_confidence_score must be in [0, 1].'
            )
        if self.local_density_contrast_min_ratio <= 0.0:
            raise ValueError(
                'p30_local_density_contrast_min_ratio must be positive.'
            )
        if self.motion_regularity_min_bins < 2:
            raise ValueError(
                'p30_motion_regularity_min_bins must be at least 2.'
            )
        if self.motion_regularity_max_residual <= 0.0:
            raise ValueError(
                'p30_motion_regularity_max_residual must be positive.'
            )
        if self.min_risk_criteria < 1 or self.min_risk_criteria > 3:
            raise ValueError('p30_min_risk_criteria must be in {1, 2, 3}.')
        if not 0.0 <= self.max_suppression_fraction <= 1.0:
            raise ValueError(
                'p30_max_suppression_fraction must be in [0, 1].'
            )

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(
                getattr(cfg, 'p30_density_component_filter_enabled', False)
            ),
            spatial_radius=int(getattr(cfg, 'p30_spatial_radius', 2)),
            temporal_bin_size=int(
                getattr(
                    cfg,
                    'p30_temporal_bin_size',
                    getattr(cfg, 'pd_detT', 50),
                )
            ),
            temporal_radius_bins=int(
                getattr(cfg, 'p30_temporal_radius_bins', 1)
            ),
            min_component_events=int(
                getattr(cfg, 'p30_min_component_events', 3)
            ),
            min_component_duration_bins=int(
                getattr(cfg, 'p30_min_component_duration_bins', 1)
            ),
            preserve_high_confidence_score=float(
                getattr(cfg, 'p30_preserve_high_confidence_score', 0.95)
            ),
            local_density_contrast_enabled=_as_bool(
                getattr(cfg, 'p30_local_density_contrast_enabled', True)
            ),
            local_density_contrast_min_ratio=float(
                getattr(cfg, 'p30_local_density_contrast_min_ratio', 0.50)
            ),
            motion_regularity_enabled=_as_bool(
                getattr(cfg, 'p30_motion_regularity_enabled', True)
            ),
            motion_regularity_min_bins=int(
                getattr(cfg, 'p30_motion_regularity_min_bins', 4)
            ),
            motion_regularity_max_residual=float(
                getattr(cfg, 'p30_motion_regularity_max_residual', 3.0)
            ),
            min_risk_criteria=int(getattr(cfg, 'p30_min_risk_criteria', 2)),
            max_suppression_fraction=float(
                getattr(cfg, 'p30_max_suppression_fraction', 0.20)
            ),
        )


@dataclass(frozen=True)
class P31MotionAwareRecoveryConfig:
    enabled: bool = False
    candidate_floor: float = 0.53
    spatial_radius: int = 5
    temporal_bin_size: int = 50
    max_link_distance: float = 8.0
    extrapolation_search_radius: float = 12.0
    max_gap_bins: int = 2
    min_track_bins: int = 4
    min_seed_components: int = 1
    max_events_per_component: int = 1
    max_recoveries_per_video: int = 0
    velocity_history_bins: int = 2

    def __post_init__(self):
        if not 0.0 <= self.candidate_floor <= 1.0:
            raise ValueError('p31_candidate_floor must be in [0, 1].')
        if self.spatial_radius < 0:
            raise ValueError('p31_spatial_radius must be non-negative.')
        if self.temporal_bin_size <= 0:
            raise ValueError('p31_temporal_bin_size must be positive.')
        if self.max_link_distance < 0.0:
            raise ValueError('p31_max_link_distance must be non-negative.')
        if self.extrapolation_search_radius <= 0.0:
            raise ValueError(
                'p31_extrapolation_search_radius must be positive.'
            )
        if self.max_gap_bins < 1:
            raise ValueError('p31_max_gap_bins must be at least 1.')
        if self.min_track_bins < 2:
            raise ValueError('p31_min_track_bins must be at least 2.')
        if self.min_seed_components < 1:
            raise ValueError('p31_min_seed_components must be positive.')
        if self.max_events_per_component < 1:
            raise ValueError('p31_max_events_per_component must be positive.')
        if self.max_recoveries_per_video < 0:
            raise ValueError(
                'p31_max_recoveries_per_video must be non-negative.'
            )
        if self.velocity_history_bins < 2:
            raise ValueError(
                'p31_velocity_history_bins must be at least 2.'
            )

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(
                getattr(cfg, 'p31_motion_aware_recovery_enabled', False)
            ),
            candidate_floor=float(getattr(cfg, 'p31_candidate_floor', 0.53)),
            spatial_radius=int(getattr(cfg, 'p31_spatial_radius', 5)),
            temporal_bin_size=int(
                getattr(
                    cfg,
                    'p31_temporal_bin_size',
                    getattr(cfg, 'pd_detT', 50),
                )
            ),
            max_link_distance=float(
                getattr(cfg, 'p31_max_link_distance', 8.0)
            ),
            extrapolation_search_radius=float(
                getattr(cfg, 'p31_extrapolation_search_radius', 12.0)
            ),
            max_gap_bins=int(getattr(cfg, 'p31_max_gap_bins', 2)),
            min_track_bins=int(getattr(cfg, 'p31_min_track_bins', 4)),
            min_seed_components=int(
                getattr(cfg, 'p31_min_seed_components', 1)
            ),
            max_events_per_component=int(
                getattr(cfg, 'p31_max_events_per_component', 1)
            ),
            max_recoveries_per_video=int(
                getattr(cfg, 'p31_max_recoveries_per_video', 0)
            ),
            velocity_history_bins=int(
                getattr(cfg, 'p31_velocity_history_bins', 2)
            ),
        )


@dataclass(frozen=True)
class P32TrackQualityBonusConfig:
    enabled: bool = False
    candidate_floor: float = 0.30
    spatial_radius: int = 2
    temporal_bin_size: int = 50
    max_link_distance: float = 8.0
    max_gap_bins: int = 2
    min_track_bins: int = 3
    min_seed_components: int = 1
    bonus: float = 0.02
    max_score_cap: float = 0.98
    max_motion_residual: float = 4.0
    velocity_history_bins: int = 2

    def __post_init__(self):
        if not 0.0 <= self.candidate_floor <= 1.0:
            raise ValueError('p32_candidate_floor must be in [0, 1].')
        if self.spatial_radius < 0:
            raise ValueError('p32_spatial_radius must be non-negative.')
        if self.temporal_bin_size <= 0:
            raise ValueError('p32_temporal_bin_size must be positive.')
        if self.max_link_distance < 0.0:
            raise ValueError('p32_max_link_distance must be non-negative.')
        if self.max_gap_bins < 1:
            raise ValueError('p32_max_gap_bins must be at least 1.')
        if self.min_track_bins < 2:
            raise ValueError('p32_min_track_bins must be at least 2.')
        if self.min_seed_components < 1:
            raise ValueError('p32_min_seed_components must be positive.')
        if self.bonus <= 0.0:
            raise ValueError('p32_bonus must be positive.')
        if not 0.0 < self.max_score_cap <= 1.0:
            raise ValueError('p32_max_score_cap must be in (0, 1].')
        if self.max_motion_residual <= 0.0:
            raise ValueError('p32_max_motion_residual must be positive.')
        if self.velocity_history_bins < 2:
            raise ValueError('p32_velocity_history_bins must be at least 2.')

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(
                getattr(cfg, 'p32_track_quality_bonus_enabled', False)
            ),
            candidate_floor=float(getattr(cfg, 'p32_candidate_floor', 0.30)),
            spatial_radius=int(getattr(cfg, 'p32_spatial_radius', 2)),
            temporal_bin_size=int(
                getattr(
                    cfg,
                    'p32_temporal_bin_size',
                    getattr(cfg, 'pd_detT', 50),
                )
            ),
            max_link_distance=float(
                getattr(cfg, 'p32_max_link_distance', 8.0)
            ),
            max_gap_bins=int(getattr(cfg, 'p32_max_gap_bins', 2)),
            min_track_bins=int(getattr(cfg, 'p32_min_track_bins', 3)),
            min_seed_components=int(
                getattr(cfg, 'p32_min_seed_components', 1)
            ),
            bonus=float(getattr(cfg, 'p32_bonus', 0.02)),
            max_score_cap=float(getattr(cfg, 'p32_max_score_cap', 0.98)),
            max_motion_residual=float(
                getattr(cfg, 'p32_max_motion_residual', 4.0)
            ),
            velocity_history_bins=int(
                getattr(cfg, 'p32_velocity_history_bins', 2)
            ),
        )


@dataclass
class P30DensityComponentFilterStats:
    enabled: bool
    input_positive_events: int = 0
    output_positive_events: int = 0
    component_count: int = 0
    risky_components: int = 0
    removed_components: int = 0
    preserved_components: int = 0
    removed_positive_events: int = 0

    def merge(self, other):
        if self.enabled != other.enabled:
            raise ValueError('Cannot merge enabled and disabled P30 stats.')
        self.input_positive_events += other.input_positive_events
        self.output_positive_events += other.output_positive_events
        self.component_count += other.component_count
        self.risky_components += other.risky_components
        self.removed_components += other.removed_components
        self.preserved_components += other.preserved_components
        self.removed_positive_events += other.removed_positive_events

    def summary(self):
        if not self.enabled:
            return 'disabled (predictions unchanged)'
        return (
            'enabled, positive events: {} -> {}; components: {} kept / {} '
            'removed ({} risky, {} preserved)'
        ).format(
            self.input_positive_events,
            self.output_positive_events,
            self.component_count - self.removed_components,
            self.removed_components,
            self.risky_components,
            self.preserved_components,
        )


@dataclass
class P31MotionAwareRecoveryStats:
    enabled: bool
    input_positive_events: int = 0
    output_positive_events: int = 0
    candidate_components: int = 0
    track_count: int = 0
    eligible_tracks: int = 0
    recovered_components: int = 0
    recovered_events: int = 0

    def merge(self, other):
        if self.enabled != other.enabled:
            raise ValueError('Cannot merge enabled and disabled P31 stats.')
        self.input_positive_events += other.input_positive_events
        self.output_positive_events += other.output_positive_events
        self.candidate_components += other.candidate_components
        self.track_count += other.track_count
        self.eligible_tracks += other.eligible_tracks
        self.recovered_components += other.recovered_components
        self.recovered_events += other.recovered_events

    def summary(self):
        if not self.enabled:
            return 'disabled (predictions unchanged)'
        return (
            'enabled, positive events: {} -> {}; candidate components: {}; '
            'eligible tracks: {} / {}; recovered: {} components / {} events'
        ).format(
            self.input_positive_events,
            self.output_positive_events,
            self.candidate_components,
            self.eligible_tracks,
            self.track_count,
            self.recovered_components,
            self.recovered_events,
        )


@dataclass
class P32TrackQualityBonusStats:
    enabled: bool
    input_positive_events: int = 0
    output_positive_events: int = 0
    track_count: int = 0
    eligible_tracks: int = 0
    boosted_events: int = 0
    newly_positive_events: int = 0

    def merge(self, other):
        if self.enabled != other.enabled:
            raise ValueError('Cannot merge enabled and disabled P32 stats.')
        self.input_positive_events += other.input_positive_events
        self.output_positive_events += other.output_positive_events
        self.track_count += other.track_count
        self.eligible_tracks += other.eligible_tracks
        self.boosted_events += other.boosted_events
        self.newly_positive_events += other.newly_positive_events

    def summary(self):
        if not self.enabled:
            return 'disabled (predictions unchanged)'
        return (
            'enabled, positive events: {} -> {}; tracks: {}; eligible: {}; '
            'boosted events: {}; newly positive: {}'
        ).format(
            self.input_positive_events,
            self.output_positive_events,
            self.track_count,
            self.eligible_tracks,
            self.boosted_events,
            self.newly_positive_events,
        )


def _fit_velocity(bin_values, centroids):
    """Fit a constant-velocity model to bin/centroid pairs."""
    bin_values = np.asarray(bin_values, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    if bin_values.shape[0] < 2 or centroids.shape[0] < 2:
        return None
    design = np.column_stack((np.ones_like(bin_values), bin_values))
    velocity_x, _ = np.linalg.lstsq(
        design,
        centroids[:, 0],
        rcond=None,
    )[:2]
    velocity_y, _ = np.linalg.lstsq(
        design,
        centroids[:, 1],
        rcond=None,
    )[:2]
    return float(velocity_x[1]), float(velocity_y[1])


def _linear_residual(bin_values, centroids):
    bin_values = np.asarray(bin_values, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    if bin_values.shape[0] < 2:
        return 0.0
    design = np.column_stack((np.ones_like(bin_values), bin_values))
    x_coeff, _, _, _ = np.linalg.lstsq(
        design,
        centroids[:, 0],
        rcond=None,
    )
    y_coeff, _, _, _ = np.linalg.lstsq(
        design,
        centroids[:, 1],
        rcond=None,
    )
    predicted = np.column_stack((design @ x_coeff, design @ y_coeff))
    residual = np.linalg.norm(centroids - predicted, axis=1)
    return float(residual.mean())


def _build_positive_components(
    positive_indices,
    coordinates,
    config,
):
    temporal_bins = np.floor_divide(
        coordinates[:, 2],
        config.temporal_bin_size,
    )
    components = []
    for temporal_bin in np.unique(temporal_bins):
        bin_mask = temporal_bins == temporal_bin
        bin_positive_indices = positive_indices[bin_mask]
        bin_coordinates = coordinates[bin_mask]
        bin_components = _spatial_components(
            bin_coordinates,
            bin_positive_indices,
            config.spatial_radius,
        )
        for component in bin_components:
            component['temporal_bin'] = int(temporal_bin)
        components.extend(bin_components)
    return components


def _component_features(component, predictions, config, video_density):
    indices = np.asarray(component['event_indices'], dtype=np.int64)
    scores = predictions[indices]
    coordinates = component.get('coordinates')
    if coordinates is None:
        raise ValueError('Component coordinates were not populated.')
    duration = component['duration_bins']
    bins_with_events = component['bin_count']
    fill_ratio = bins_with_events / max(duration, 1)
    area = max(component['bbox_area'], 1)
    component_density = component['event_count'] / max(area * duration, 1)
    density_ratio = component_density / max(video_density, 1e-12)
    density_deficit = max(
        0.0,
        1.0 - density_ratio / config.local_density_contrast_min_ratio,
    )
    motion_residual = component.get('motion_residual', 0.0)
    motion_ratio = min(
        2.0,
        motion_residual / max(config.motion_regularity_max_residual, 1e-6),
    )
    fill_deficit = max(0.0, 0.60 - fill_ratio) / 0.60
    risk_score = 0.50 * motion_ratio + 0.30 * density_deficit + 0.20 * fill_deficit
    criteria = 0
    if config.local_density_contrast_enabled and density_ratio < (
        config.local_density_contrast_min_ratio
    ):
        criteria += 1
    if (
        config.motion_regularity_enabled
        and duration >= config.motion_regularity_min_bins
        and motion_residual > config.motion_regularity_max_residual
    ):
        criteria += 1
    if fill_ratio < 0.5 and duration >= 2:
        criteria += 1
    return {
        'risk_score': float(risk_score),
        'criteria': criteria,
        'max_score': float(scores.max()),
        'high_confidence': bool(
            scores.max() >= config.preserve_high_confidence_score
        ),
        'density_ratio': float(density_ratio),
        'motion_residual': float(motion_residual),
        'fill_ratio': float(fill_ratio),
    }


def _filter_one_video_by_density(predictions, coordinates, threshold, config):
    event_count = predictions.shape[0]
    stats = P30DensityComponentFilterStats(
        enabled=True,
        input_positive_events=int((predictions >= threshold).sum()),
    )
    positive_indices = np.flatnonzero(predictions >= threshold)
    if positive_indices.size == 0 or event_count == 0:
        stats.output_positive_events = stats.input_positive_events
        return np.zeros(event_count, dtype=bool), stats

    positive_coordinates = coordinates[positive_indices]
    components = _build_positive_components(
        positive_indices,
        positive_coordinates,
        config,
    )
    if not components:
        stats.output_positive_events = stats.input_positive_events
        return np.zeros(event_count, dtype=bool), stats

    total_bins = int(
        np.floor_divide(coordinates[:, 2].max(), config.temporal_bin_size) + 1
    )
    max_x = int(coordinates[:, 0].max()) + 1
    max_y = int(coordinates[:, 1].max()) + 1
    video_density = event_count / max(max_x * max_y * total_bins, 1)

    component_records = []
    for component in components:
        indices = np.asarray(component['event_indices'], dtype=np.int64)
        component['coordinates'] = coordinates[indices]
        component['duration_bins'] = 1
        component['bin_count'] = 1
        component['bbox_area'] = int(component['event_count'])
        component['motion_residual'] = 0.0
        component_records.append((component, indices))

    # Group spatial components across neighboring temporal bins into
    # spatiotemporal components for motion/density statistics.
    temporal_components = []
    used = np.zeros(len(component_records), dtype=bool)
    for start_index, (start_component, _) in enumerate(component_records):
        if used[start_index]:
            continue
        used[start_index] = True
        merged_indices = [np.asarray(start_component['event_indices'], dtype=np.int64)]
        merged_cells = list(start_component['cells'])
        merged_bins = [int(start_component['temporal_bin'])]
        start_coord = start_component['coordinates']
        stack = [start_index]
        while stack:
            current_index = stack.pop()
            current_component = component_records[current_index][0]
            current_bin = int(current_component['temporal_bin'])
            current_centroid = current_component['centroid']
            for other_index, (other_component, _) in enumerate(component_records):
                if used[other_index]:
                    continue
                bin_difference = abs(
                    int(other_component['temporal_bin']) - current_bin
                )
                if bin_difference > config.temporal_radius_bins:
                    continue
                # Use a generous spatial link distance for temporal grouping.
                # The official diagnostics show p99 target motion is about
                # 8.3 px/bin, so this needs to exceed that value while still
                # preventing completely unrelated components from merging.
                if np.linalg.norm(
                    other_component['centroid'] - current_centroid
                ) > max(12.0, config.spatial_radius * 3):
                    continue
                used[other_index] = True
                merged_indices.append(
                    np.asarray(other_component['event_indices'], dtype=np.int64)
                )
                merged_cells.extend(other_component['cells'])
                merged_bins.append(int(other_component['temporal_bin']))
                stack.append(other_index)

        merged_indices = np.concatenate(merged_indices)
        merged_cells = np.asarray(merged_cells, dtype=np.int64)
        merged_bins = np.asarray(merged_bins, dtype=np.int64)
        unique_bins = np.unique(merged_bins)
        bbox_area = int(merged_cells.max() + 1) if merged_cells.size else 1
        # Recompute a proper spatial bbox from raw coordinates.
        merged_coords = coordinates[merged_indices]
        min_xy = merged_coords[:, :2].min(axis=0)
        max_xy = merged_coords[:, :2].max(axis=0)
        bbox_area = int(
            max(1, max_xy[0] - min_xy[0] + 1)
            * max(1, max_xy[1] - min_xy[1] + 1)
        )
        duration = int(unique_bins.max() - unique_bins.min() + 1)
        per_bin_centroids = []
        per_bin_values = []
        for temporal_bin in unique_bins:
            bin_coords = merged_coords[
                np.floor_divide(
                    merged_coords[:, 2],
                    config.temporal_bin_size,
                )
                == temporal_bin
            ]
            per_bin_centroids.append(bin_coords[:, :2].mean(axis=0))
            per_bin_values.append(float(temporal_bin))
        motion_residual = 0.0
        if (
            config.motion_regularity_enabled
            and len(per_bin_values) >= config.motion_regularity_min_bins
        ):
            motion_residual = _linear_residual(
                np.asarray(per_bin_values),
                np.asarray(per_bin_centroids),
            )
        temporal_components.append(
            {
                'event_indices': merged_indices,
                'coordinates': merged_coords,
                'event_count': int(merged_indices.size),
                'duration_bins': duration,
                'bin_count': int(unique_bins.size),
                'bbox_area': bbox_area,
                'motion_residual': motion_residual,
            }
        )

    risky = []
    for component in temporal_components:
        features = _component_features(
            component,
            predictions,
            config,
            video_density,
        )
        component['features'] = features
        if features['criteria'] >= config.min_risk_criteria:
            risky.append(component)

    stats.component_count = len(temporal_components)
    stats.risky_components = len(risky)
    max_removals = int(
        np.ceil(len(temporal_components) * config.max_suppression_fraction)
    )
    risky.sort(
        key=lambda component: component['features']['risk_score'],
        reverse=True,
    )
    selected_for_removal = []
    for component in risky:
        if component['features']['high_confidence']:
            stats.preserved_components += 1
            continue
        if len(selected_for_removal) >= max_removals:
            break
        selected_for_removal.append(component)

    removal_mask = np.zeros(event_count, dtype=bool)
    for component in selected_for_removal:
        removal_mask[np.asarray(component['event_indices'], dtype=np.int64)] = True

    stats.removed_components = len(selected_for_removal)
    stats.removed_positive_events = int(removal_mask.sum())
    stats.output_positive_events = stats.input_positive_events - stats.removed_positive_events
    return removal_mask, stats


class P30DensityComponentFilter:
    def __init__(self, config, prediction_threshold=0.9):
        if not 0.0 <= prediction_threshold <= 1.0:
            raise ValueError('prediction_threshold must be in [0, 1].')
        self.config = config
        self.prediction_threshold = float(prediction_threshold)

    @classmethod
    def from_cfg(cls, cfg, prediction_threshold=0.9):
        return cls(
            P30DensityComponentFilterConfig.from_cfg(cfg),
            prediction_threshold,
        )

    @property
    def enabled(self):
        return self.config.enabled

    def new_stats(self):
        return P30DensityComponentFilterStats(enabled=self.enabled)

    def describe(self):
        if not self.enabled:
            return 'disabled'
        return (
            'enabled (spatial_radius={}, temporal_bin_size={}, '
            'min_component_events={}, preserve_high_confidence_score={}, '
            'density_contrast={}, motion_regularity={}, '
            'min_risk_criteria={}, max_suppression_fraction={})'
        ).format(
            self.config.spatial_radius,
            self.config.temporal_bin_size,
            self.config.min_component_events,
            self.config.preserve_high_confidence_score,
            self.config.local_density_contrast_enabled,
            self.config.motion_regularity_enabled,
            self.config.min_risk_criteria,
            self.config.max_suppression_fraction,
        )

    def apply(self, predictions, locations):
        if not self.enabled:
            stats = P30DensityComponentFilterStats(enabled=False)
            positive_count = int(
                (predictions.reshape(-1) >= self.prediction_threshold).sum()
            )
            stats.input_positive_events = positive_count
            stats.output_positive_events = positive_count
            return predictions, stats

        import torch

        flattened = predictions.reshape(-1)
        if flattened.numel() != locations.shape[0]:
            raise ValueError(
                'Prediction and location counts do not match: {} and {}.'.format(
                    flattened.numel(),
                    locations.shape[0],
                )
            )
        prediction_values = flattened.detach().cpu().numpy().astype(np.float64)
        location_values = locations.detach().cpu().numpy()
        stats = P30DensityComponentFilterStats(enabled=True)
        batch_ids = location_values[:, 0].astype(np.int64, copy=False)
        removal_mask = np.zeros(prediction_values.shape[0], dtype=bool)
        for batch_id in np.unique(batch_ids):
            video_mask = batch_ids == batch_id
            video_removal, video_stats = _filter_one_video_by_density(
                prediction_values[video_mask],
                location_values[video_mask, 1:4].astype(
                    np.int64,
                    copy=False,
                ),
                self.prediction_threshold,
                self.config,
            )
            removal_mask[video_mask] = video_removal
            stats.merge(video_stats)
        if not removal_mask.any():
            return predictions, stats
        filtered = flattened.clone()
        filtered[torch.from_numpy(removal_mask).to(device=filtered.device)] = 0.0
        return filtered.reshape_as(predictions), stats


def _recover_one_video_motion(predictions, coordinates, threshold, config):
    event_count = predictions.shape[0]
    stats = P31MotionAwareRecoveryStats(
        enabled=True,
        input_positive_events=int((predictions >= threshold).sum()),
    )
    recovery_mask = np.zeros(event_count, dtype=bool)
    seed_mask = predictions >= threshold
    weak_mask = (predictions >= config.candidate_floor) & ~seed_mask
    active_mask = seed_mask | weak_mask
    if not seed_mask.any() or not weak_mask.any() or not active_mask.any():
        stats.output_positive_events = stats.input_positive_events
        return recovery_mask, stats

    active_indices = np.flatnonzero(active_mask)
    active_coordinates = coordinates[active_indices]
    temporal_bins = np.floor_divide(
        active_coordinates[:, 2],
        config.temporal_bin_size,
    )
    tracks = []
    velocity_points_by_track = []

    for temporal_bin in np.unique(temporal_bins):
        bin_active_indices = np.flatnonzero(temporal_bins == temporal_bin)
        components = _spatial_components(
            active_coordinates[bin_active_indices],
            active_indices[bin_active_indices],
            config.spatial_radius,
        )
        for component in components:
            component['has_seed'] = bool(
                seed_mask[component['event_indices']].any()
            )
            component_scores = predictions[component['event_indices']]
            component['sorted_event_indices'] = component['event_indices'][
                np.argsort(component_scores)[::-1]
            ]
        stats.candidate_components += len(components)

        links = []
        for track_index, track in enumerate(tracks):
            bin_difference = int(temporal_bin - track['last_bin'])
            if bin_difference <= 0:
                continue
            velocity = track.get('velocity')
            for component_index, component in enumerate(components):
                last_distance = float(
                    np.linalg.norm(component['centroid'] - track['centroid'])
                )
                distance = last_distance
                use_extrapolation = False
                if bin_difference <= config.max_gap_bins:
                    if last_distance <= config.max_link_distance:
                        links.append((last_distance, track_index, component_index, False))
                if velocity is not None:
                    predicted = track['centroid'] + np.asarray(velocity) * bin_difference
                    predicted_distance = float(
                        np.linalg.norm(component['centroid'] - predicted)
                    )
                    if predicted_distance <= config.extrapolation_search_radius:
                        links.append(
                            (
                                predicted_distance,
                                track_index,
                                component_index,
                                True,
                            )
                        )
        assigned_tracks = set()
        assigned_components = set()
        for _, track_index, component_index, _ in sorted(links):
            if track_index in assigned_tracks or component_index in assigned_components:
                continue
            track = tracks[track_index]
            component = components[component_index]
            track['components'].append(component)
            track['frame_count'] += 1
            track['has_seed'] = track['has_seed'] or component['has_seed']
            track['centroid'] = component['centroid']
            track['last_bin'] = int(temporal_bin)
            velocity_points_by_track[track_index].append(
                (int(temporal_bin), component['centroid'])
            )
            if len(velocity_points_by_track[track_index]) >= 2:
                recent_points = velocity_points_by_track[track_index][
                    -int(config.velocity_history_bins):
                ]
                track['velocity'] = _fit_velocity(
                    [point[0] for point in recent_points],
                    [point[1] for point in recent_points],
                )
            assigned_tracks.add(track_index)
            assigned_components.add(component_index)

        for component_index, component in enumerate(components):
            if component_index in assigned_components:
                continue
            track = {
                'components': [component],
                'frame_count': 1,
                'has_seed': component['has_seed'],
                'centroid': component['centroid'],
                'last_bin': int(temporal_bin),
                'velocity': None,
            }
            tracks.append(track)
            velocity_points_by_track.append(
                [(int(temporal_bin), component['centroid'])]
            )

    stats.track_count = len(tracks)
    total_recovered = 0
    for track in tracks:
        seed_component_count = int(
            sum(1 for component in track['components'] if component['has_seed'])
        )
        if (
            not track['has_seed']
            or track['frame_count'] < config.min_track_bins
            or seed_component_count < config.min_seed_components
        ):
            continue
        stats.eligible_tracks += 1
        for component in track['components']:
            if component['has_seed']:
                continue
            if (
                config.max_recoveries_per_video
                and total_recovered >= config.max_recoveries_per_video
            ):
                break
            recovered_indices = component['sorted_event_indices'][
                : config.max_events_per_component
            ]
            if recovered_indices.size == 0:
                continue
            recovery_mask[recovered_indices] = True
            stats.recovered_components += 1
            stats.recovered_events += int(recovered_indices.size)
            total_recovered += int(recovered_indices.size)
        if (
            config.max_recoveries_per_video
            and total_recovered >= config.max_recoveries_per_video
        ):
            break

    stats.output_positive_events = stats.input_positive_events + stats.recovered_events
    return recovery_mask, stats


class P31MotionAwareTrackRecovery:
    def __init__(self, config, prediction_threshold=0.9):
        if not 0.0 <= prediction_threshold <= 1.0:
            raise ValueError('prediction_threshold must be in [0, 1].')
        self.config = config
        self.prediction_threshold = float(prediction_threshold)

    @classmethod
    def from_cfg(cls, cfg, prediction_threshold=0.9):
        return cls(
            P31MotionAwareRecoveryConfig.from_cfg(cfg),
            prediction_threshold,
        )

    @property
    def enabled(self):
        return self.config.enabled

    def new_stats(self):
        return P31MotionAwareRecoveryStats(enabled=self.enabled)

    def describe(self):
        if not self.enabled:
            return 'disabled'
        return (
            'enabled (candidate_floor={}, spatial_radius={}, '
            'temporal_bin_size={}, max_link_distance={}, '
            'extrapolation_search_radius={}, max_gap_bins={}, '
            'min_track_bins={}, min_seed_components={}, '
            'max_events_per_component={}, max_recoveries_per_video={})'
        ).format(
            self.config.candidate_floor,
            self.config.spatial_radius,
            self.config.temporal_bin_size,
            self.config.max_link_distance,
            self.config.extrapolation_search_radius,
            self.config.max_gap_bins,
            self.config.min_track_bins,
            self.config.min_seed_components,
            self.config.max_events_per_component,
            self.config.max_recoveries_per_video,
        )

    def apply(self, predictions, locations):
        if not self.enabled:
            stats = P31MotionAwareRecoveryStats(enabled=False)
            positive_count = int(
                (predictions.reshape(-1) >= self.prediction_threshold).sum()
            )
            stats.input_positive_events = positive_count
            stats.output_positive_events = positive_count
            return predictions, stats

        import torch

        flattened = predictions.reshape(-1)
        if flattened.numel() != locations.shape[0]:
            raise ValueError(
                'Prediction and location counts do not match: {} and {}.'.format(
                    flattened.numel(),
                    locations.shape[0],
                )
            )
        prediction_values = flattened.detach().cpu().numpy().astype(np.float64)
        location_values = locations.detach().cpu().numpy()
        stats = P31MotionAwareRecoveryStats(enabled=True)
        batch_ids = location_values[:, 0].astype(np.int64, copy=False)
        recovery_mask = np.zeros(prediction_values.shape[0], dtype=bool)
        for batch_id in np.unique(batch_ids):
            video_mask = batch_ids == batch_id
            video_recovery, video_stats = _recover_one_video_motion(
                prediction_values[video_mask],
                location_values[video_mask, 1:4].astype(
                    np.int64,
                    copy=False,
                ),
                self.prediction_threshold,
                self.config,
            )
            recovery_mask[video_mask] = video_recovery
            stats.merge(video_stats)
        if not recovery_mask.any():
            return predictions, stats
        recovered = flattened.clone()
        recovered[torch.from_numpy(recovery_mask).to(device=recovered.device)] = (
            self.prediction_threshold
        )
        return recovered.reshape_as(predictions), stats


def _bonus_one_video(predictions, coordinates, threshold, config):
    """Return a per-event score bonus for seed-supported smooth tracks."""
    event_count = predictions.shape[0]
    stats = P32TrackQualityBonusStats(
        enabled=True,
        input_positive_events=int((predictions >= threshold).sum()),
    )
    bonus_values = np.zeros(event_count, dtype=np.float64)
    seed_mask = predictions >= threshold
    candidate_mask = predictions >= config.candidate_floor
    if not seed_mask.any() or not candidate_mask.any():
        stats.output_positive_events = stats.input_positive_events
        return bonus_values, stats

    active_indices = np.flatnonzero(candidate_mask)
    active_coordinates = coordinates[active_indices]
    temporal_bins = np.floor_divide(
        active_coordinates[:, 2],
        config.temporal_bin_size,
    )
    tracks = []
    velocity_points_by_track = []

    for temporal_bin in np.unique(temporal_bins):
        bin_active_indices = np.flatnonzero(temporal_bins == temporal_bin)
        components = _spatial_components(
            active_coordinates[bin_active_indices],
            active_indices[bin_active_indices],
            config.spatial_radius,
        )
        for component in components:
            component['has_seed'] = bool(
                seed_mask[component['event_indices']].any()
            )

        links = []
        for track_index, track in enumerate(tracks):
            bin_difference = int(temporal_bin - track['last_bin'])
            if bin_difference <= 0:
                continue
            velocity = track.get('velocity')
            for component_index, component in enumerate(components):
                last_distance = float(
                    np.linalg.norm(component['centroid'] - track['centroid'])
                )
                if bin_difference <= config.max_gap_bins:
                    if last_distance <= config.max_link_distance:
                        links.append((last_distance, track_index, component_index))
                if velocity is not None:
                    predicted = (
                        track['centroid'] + np.asarray(velocity) * bin_difference
                    )
                    predicted_distance = float(
                        np.linalg.norm(component['centroid'] - predicted)
                    )
                    if predicted_distance <= config.max_link_distance * 1.5:
                        links.append(
                            (predicted_distance, track_index, component_index)
                        )

        assigned_tracks = set()
        assigned_components = set()
        for _, track_index, component_index in sorted(links):
            if track_index in assigned_tracks or component_index in assigned_components:
                continue
            track = tracks[track_index]
            component = components[component_index]
            track['components'].append(component)
            track['frame_count'] += 1
            track['has_seed'] = track['has_seed'] or component['has_seed']
            track['centroid'] = component['centroid']
            track['last_bin'] = int(temporal_bin)
            velocity_points_by_track[track_index].append(
                (int(temporal_bin), component['centroid'])
            )
            if len(velocity_points_by_track[track_index]) >= 2:
                recent_points = velocity_points_by_track[track_index][
                    -int(config.velocity_history_bins):
                ]
                track['velocity'] = _fit_velocity(
                    [point[0] for point in recent_points],
                    [point[1] for point in recent_points],
                )
            assigned_tracks.add(track_index)
            assigned_components.add(component_index)

        for component_index, component in enumerate(components):
            if component_index in assigned_components:
                continue
            track = {
                'components': [component],
                'frame_count': 1,
                'has_seed': component['has_seed'],
                'centroid': component['centroid'],
                'last_bin': int(temporal_bin),
                'velocity': None,
            }
            tracks.append(track)
            velocity_points_by_track.append(
                [(int(temporal_bin), component['centroid'])]
            )

    stats.track_count = len(tracks)
    boosted_events = 0
    for track_index, track in enumerate(tracks):
        seed_component_count = int(
            sum(1 for component in track['components'] if component['has_seed'])
        )
        if (
            not track['has_seed']
            or track['frame_count'] < config.min_track_bins
            or seed_component_count < config.min_seed_components
        ):
            continue
        points = velocity_points_by_track[track_index]
        residual = _linear_residual(
            np.asarray([point[0] for point in points]),
            np.asarray([point[1] for point in points]),
        )
        if residual > config.max_motion_residual:
            continue
        stats.eligible_tracks += 1
        for component in track['components']:
            indices = np.asarray(component['event_indices'], dtype=np.int64)
            bonus_values[indices] = config.bonus
            boosted_events += int(indices.size)
    stats.boosted_events = boosted_events

    output_positive = int(
        (
            np.minimum(config.max_score_cap, predictions + bonus_values)
            >= threshold
        ).sum()
    )
    stats.newly_positive_events = max(
        0,
        output_positive - stats.input_positive_events,
    )
    stats.output_positive_events = output_positive
    return bonus_values, stats


class P32TrackQualityBonus:
    def __init__(self, config, prediction_threshold=0.9):
        if not 0.0 <= prediction_threshold <= 1.0:
            raise ValueError('prediction_threshold must be in [0, 1].')
        self.config = config
        self.prediction_threshold = float(prediction_threshold)

    @classmethod
    def from_cfg(cls, cfg, prediction_threshold=0.9):
        return cls(
            P32TrackQualityBonusConfig.from_cfg(cfg),
            prediction_threshold,
        )

    @property
    def enabled(self):
        return self.config.enabled

    def new_stats(self):
        return P32TrackQualityBonusStats(enabled=self.enabled)

    def describe(self):
        if not self.enabled:
            return 'disabled'
        return (
            'enabled (candidate_floor={}, spatial_radius={}, '
            'temporal_bin_size={}, max_link_distance={}, max_gap_bins={}, '
            'min_track_bins={}, min_seed_components={}, bonus={}, '
            'max_score_cap={}, max_motion_residual={})'
        ).format(
            self.config.candidate_floor,
            self.config.spatial_radius,
            self.config.temporal_bin_size,
            self.config.max_link_distance,
            self.config.max_gap_bins,
            self.config.min_track_bins,
            self.config.min_seed_components,
            self.config.bonus,
            self.config.max_score_cap,
            self.config.max_motion_residual,
        )

    def apply(self, predictions, locations):
        if not self.enabled:
            stats = P32TrackQualityBonusStats(enabled=False)
            positive_count = int(
                (predictions.reshape(-1) >= self.prediction_threshold).sum()
            )
            stats.input_positive_events = positive_count
            stats.output_positive_events = positive_count
            return predictions, stats

        import torch

        flattened = predictions.reshape(-1)
        if flattened.numel() != locations.shape[0]:
            raise ValueError(
                'Prediction and location counts do not match: {} and {}.'.format(
                    flattened.numel(),
                    locations.shape[0],
                )
            )
        prediction_values = flattened.detach().cpu().numpy().astype(np.float64)
        location_values = locations.detach().cpu().numpy()
        stats = P32TrackQualityBonusStats(enabled=True)
        bonus_values = np.zeros(prediction_values.shape[0], dtype=np.float64)
        batch_ids = location_values[:, 0].astype(np.int64, copy=False)
        for batch_id in np.unique(batch_ids):
            video_mask = batch_ids == batch_id
            video_bonus, video_stats = _bonus_one_video(
                prediction_values[video_mask],
                location_values[video_mask, 1:4].astype(
                    np.int64,
                    copy=False,
                ),
                self.prediction_threshold,
                self.config,
            )
            bonus_values[video_mask] = video_bonus
            stats.merge(video_stats)
        if not bonus_values.any():
            return predictions, stats
        boosted = flattened.clone()
        boosted_values = np.minimum(
            self.config.max_score_cap,
            prediction_values + bonus_values,
        )
        boosted.copy_(torch.from_numpy(boosted_values).to(device=boosted.device))
        return boosted.reshape_as(predictions), stats


@dataclass
class AdaptiveRound4Stats:
    filter_stats: P30DensityComponentFilterStats
    recovery_stats: P31MotionAwareRecoveryStats
    bonus_stats: P32TrackQualityBonusStats

    def merge(self, other):
        self.filter_stats.merge(other.filter_stats)
        self.recovery_stats.merge(other.recovery_stats)
        self.bonus_stats.merge(other.bonus_stats)

    def summary(self):
        return (
            'P30 density-component filter: {}; P31 motion-aware recovery: {}; '
            'P32 track-quality bonus: {}'
        ).format(
            self.filter_stats.summary(),
            self.recovery_stats.summary(),
            self.bonus_stats.summary(),
        )


class AdaptiveRound4Postprocessor:
    """Stack P30 label-free component filtering and P31 motion recovery."""

    def __init__(self, component_filter, motion_recovery, track_bonus):
        self._component_filter = component_filter
        self._motion_recovery = motion_recovery
        self._track_bonus = track_bonus

    @classmethod
    def from_cfg(cls, cfg, prediction_threshold=0.9):
        return cls(
            P30DensityComponentFilter.from_cfg(cfg, prediction_threshold),
            P31MotionAwareTrackRecovery.from_cfg(cfg, prediction_threshold),
            P32TrackQualityBonus.from_cfg(cfg, prediction_threshold),
        )

    @property
    def enabled(self):
        return (
            self._component_filter.enabled
            or self._motion_recovery.enabled
            or self._track_bonus.enabled
        )

    def new_stats(self):
        return AdaptiveRound4Stats(
            self._component_filter.new_stats(),
            self._motion_recovery.new_stats(),
            self._track_bonus.new_stats(),
        )

    def describe(self):
        return (
            'P30 density-component filter: {}; P31 motion-aware recovery: {}; '
            'P32 track-quality bonus: {}'
        ).format(
            self._component_filter.describe(),
            self._motion_recovery.describe(),
            self._track_bonus.describe(),
        )

    def apply(self, predictions, locations):
        predictions, filter_stats = self._component_filter.apply(
            predictions,
            locations,
        )
        predictions, recovery_stats = self._motion_recovery.apply(
            predictions,
            locations,
        )
        predictions, bonus_stats = self._track_bonus.apply(
            predictions,
            locations,
        )
        return predictions, AdaptiveRound4Stats(
            filter_stats,
            recovery_stats,
            bonus_stats,
        )
