#!/usr/bin/env python3
"""Apply the Round-4 adaptive post-processing patch to an evsod-main repo.

Run this script from the repository root after copying the package's
``utils/adaptive_postprocess.py`` and ``tests/test_adaptive_postprocess.py``
into the corresponding directories.  It edits:

- configs/evisseg_evuav.yaml
- test2.py
- submit_challenge2.py

All changes are additive and idempotent: the new module is imported and the
two new stages remain disabled by default.
"""

from pathlib import Path
import re
import sys


CONFIG_SNIPPET = """
  # P30: label-free density/motion component filter. It is a conservative,
  # generalizable approximation of the teammate's supervised component
  # classifier; all thresholds are fixed/observable and do not use video names.
  p30_density_component_filter_enabled: false
  p30_spatial_radius: 2
  p30_temporal_bin_size: 50
  p30_temporal_radius_bins: 1
  p30_min_component_events: 3
  p30_min_component_duration_bins: 1
  p30_preserve_high_confidence_score: 0.95
  p30_local_density_contrast_enabled: true
  p30_local_density_contrast_min_ratio: 0.50
  p30_motion_regularity_enabled: true
  p30_motion_regularity_min_bins: 4
  p30_motion_regularity_max_residual: 3.0
  p30_min_risk_criteria: 2
  p30_max_suppression_fraction: 0.20
  # P31: constant-velocity extrapolation for seed-supported weak tracks.
  # Targets high-speed, small, low-event targets that drop below threshold for
  # one or two bins. Label-free and independent of video identity.
  p31_motion_aware_recovery_enabled: false
  p31_candidate_floor: 0.53
  p31_spatial_radius: 5
  p31_temporal_bin_size: 50
  p31_max_link_distance: 8.0
  p31_extrapolation_search_radius: 12.0
  p31_max_gap_bins: 2
  p31_min_track_bins: 4
  p31_min_seed_components: 1
  p31_max_events_per_component: 1
  p31_max_recoveries_per_video: 0
  p31_velocity_history_bins: 2
  # P32: add a small score bonus to events on seed-supported, motion-regular
  # tracks. Unlike P31 it does not search for extrapolated weak components;
  # it only boosts track-consistent candidates that are already present.
  p32_track_quality_bonus_enabled: false
  p32_candidate_floor: 0.30
  p32_spatial_radius: 2
  p32_temporal_bin_size: 50
  p32_max_link_distance: 8.0
  p32_max_gap_bins: 2
  p32_min_track_bins: 3
  p32_min_seed_components: 1
  p32_bonus: 0.02
  p32_max_score_cap: 0.98
  p32_max_motion_residual: 4.0
  p32_velocity_history_bins: 2
"""

P32_SNIPPET = """
  # P32: add a small score bonus to events on seed-supported, motion-regular
  # tracks. Unlike P31 it does not search for extrapolated weak components;
  # it only boosts track-consistent candidates that are already present.
  p32_track_quality_bonus_enabled: false
  p32_candidate_floor: 0.30
  p32_spatial_radius: 2
  p32_temporal_bin_size: 50
  p32_max_link_distance: 8.0
  p32_max_gap_bins: 2
  p32_min_track_bins: 3
  p32_min_seed_components: 1
  p32_bonus: 0.02
  p32_max_score_cap: 0.98
  p32_max_motion_residual: 4.0
  p32_velocity_history_bins: 2
"""


TEST_IMPORT = "from utils.adaptive_postprocess import AdaptiveRound4Postprocessor\n"
SUBMIT_IMPORT = TEST_IMPORT

TEST_INIT = """    adaptive_postprocessor = AdaptiveRound4Postprocessor.from_cfg(
        cfg,
        PREDICTION_THRESHOLD,
    )
    adaptive_postprocess_stats = adaptive_postprocessor.new_stats()
"""

TEST_PRINT = '    print("round4 adaptive postprocessor:", adaptive_postprocessor.describe())\n'

TEST_APPLY_BLOCK = """batch_adaptive_postprocessor = (
    AdaptiveRound4Postprocessor.from_cfg(cfg, batch_threshold)
    if threshold_policy.enabled else adaptive_postprocessor
)
predictions, batch_adaptive_stats = (
    batch_adaptive_postprocessor.apply(predictions, batch["locs"])
)
adaptive_postprocess_stats.merge(batch_adaptive_stats)
"""

TEST_FINAL_PRINT = """    print(
        "round4 adaptive postprocess result:",
        adaptive_postprocess_stats.summary(),
    )
"""


def _read(path):
    return path.read_text(encoding="utf-8")


def _write(path, text):
    path.write_text(text, encoding="utf-8")


def _insert_after_unique(text, marker, insertion):
    if marker not in text:
        raise RuntimeError(f"marker not found:\n{marker}")
    if text.count(marker) != 1:
        raise RuntimeError(f"marker is not unique ({text.count(marker)}):\n{marker}")
    return text.replace(marker, marker + insertion, 1)


def _insert_after_all(text, marker, insertion):
    if marker not in text:
        raise RuntimeError(f"marker not found:\n{marker}")
    return text.replace(marker, marker + insertion)


def _insert_after_all_indented(text, marker, block):
    if marker not in text:
        raise RuntimeError(f"marker not found:\n{marker}")
    marker_no_newline = marker.rstrip("\n")
    pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)" + re.escape(marker_no_newline) + r"\n"
    )

    def replacement(match):
        indent = match.group("indent")
        indented = "".join(
            (indent + line if line.strip() else line)
            for line in block.splitlines(True)
        )
        return match.group(0) + indented

    return pattern.sub(replacement, text)


def patch_config(config_path):
    text = _read(config_path)
    if "p32_track_quality_bonus_enabled" in text:
        print("config already patched:", config_path)
        return
    if "p30_density_component_filter_enabled" in text:
        fusion_marker = "\nFUSION:\n"
        if fusion_marker in text:
            text = text.replace(fusion_marker, P32_SNIPPET + fusion_marker, 1)
        else:
            text = text.rstrip() + P32_SNIPPET + "\n"
        _write(config_path, text)
        print("patched config (P32 only):", config_path)
        return
    marker = "  p6_high_density_threshold: 0.92\n"
    if marker not in text:
        # Some evsod-main variants use a different default but the key still
        # exists; append before FUSION: as a fallback.
        fusion_marker = "\nFUSION:\n"
        if fusion_marker in text:
            text = text.replace(
                fusion_marker,
                CONFIG_SNIPPET + fusion_marker,
                1,
            )
        else:
            text = text.rstrip() + CONFIG_SNIPPET + "\n"
    else:
        text = text.replace(marker, marker + CONFIG_SNIPPET, 1)
    _write(config_path, text)
    print("patched config:", config_path)


def patch_test2(path):
    text = _read(path)
    if "AdaptiveRound4Postprocessor" in text:
        print("test2.py already patched:", path)
        return

    import_marker = "from utils.postprocess import ChallengePostprocessor\n"
    text = _insert_after_unique(text, import_marker, TEST_IMPORT)

    init_marker = "    postprocess_stats = postprocessor.new_stats()\n"
    text = _insert_after_unique(text, init_marker, TEST_INIT)

    print_marker = '    print("postprocessor:", postprocessor.describe())\n'
    text = _insert_after_unique(text, print_marker, TEST_PRINT)

    apply_marker = "postprocess_stats.merge(batch_postprocess_stats)\n"
    text = _insert_after_all_indented(text, apply_marker, TEST_APPLY_BLOCK)

    final_marker = '    print("postprocess result:", postprocess_stats.summary())\n'
    text = _insert_after_unique(text, final_marker, TEST_FINAL_PRINT)
    _write(path, text)
    print("patched test2.py:", path)


def patch_submit(path):
    text = _read(path)
    if "AdaptiveRound4Postprocessor" in text:
        print("submit_challenge2.py already patched:", path)
        return

    import_marker = "from utils.postprocess import ChallengePostprocessor\n"
    text = _insert_after_unique(text, import_marker, SUBMIT_IMPORT)

    init_marker = "    postprocess_stats = postprocessor.new_stats()\n"
    text = _insert_after_unique(text, init_marker, TEST_INIT)

    print_marker = '    print("postprocessor:", postprocessor.describe())\n'
    text = _insert_after_unique(text, print_marker, TEST_PRINT)

    apply_marker = "postprocess_stats.merge(batch_postprocess_stats)\n"
    text = _insert_after_all_indented(text, apply_marker, TEST_APPLY_BLOCK)

    final_marker = '    print("postprocess result:", postprocess_stats.summary())\n'
    text = _insert_after_unique(text, final_marker, TEST_FINAL_PRINT)
    _write(path, text)
    print("patched submit_challenge2.py:", path)


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    required = [
        repo / "utils" / "adaptive_postprocess.py",
        repo / "tests" / "test_adaptive_postprocess.py",
        repo / "configs" / "evisseg_evuav.yaml",
        repo / "test2.py",
        repo / "submit_challenge2.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("Missing required files:")
        for path in missing:
            print(" -", path)
        raise SystemExit(1)

    patch_config(required[2])
    patch_test2(required[3])
    patch_submit(required[4])
    print("Round-4 patch applied. Run `python tests/test_adaptive_postprocess.py`.")


if __name__ == "__main__":
    main()
