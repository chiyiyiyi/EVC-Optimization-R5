#!/usr/bin/env python3
"""Attach TrackQueryHead to the M26 temporal-memory stack.

Usage from the repository root:

    python apply_query_head_patch.py

It edits:

- model/temporal_memory_net.py
- train_temporal_memory.py
- utils/temporal_memory_inference.py
- configs/evisseg_evuav.yaml

The head is disabled by default, so old M26 checkpoints still load with
strict state-dict matching.
"""

from pathlib import Path
import re
import sys


CONFIG_SNIPPET = """
  # Object-level track query head (disabled by default).
  temporal_memory_track_query_head_enabled: false
  temporal_memory_track_query_num_queries: 32
  temporal_memory_track_query_hidden: 128
  temporal_memory_track_query_max_flow: 2.0
  temporal_memory_track_query_loss_weight: 0.05
  temporal_memory_track_query_warmup_epochs: 3
  temporal_memory_track_query_lr_multiplier: 1.0
"""


def _read(path):
    return path.read_text(encoding="utf-8")


def _write(path, text):
    path.write_text(text, encoding="utf-8")


def _replace_required(text, marker, replacement, label):
    if marker not in text:
        raise SystemExit(
            "Missing marker in {}:\n{}".format(label, marker)
        )
    return text.replace(marker, marker + replacement, 1)


def _insert_before_required(text, marker, block, label):
    if marker not in text:
        raise SystemExit(
            "Missing marker in {}:\n{}".format(label, marker)
        )
    return text.replace(marker, block + marker, 1)


def _insert_before_last_in_function(text, start_marker, end_marker, line_marker, block, label):
    if start_marker not in text or end_marker not in text:
        raise SystemExit(
            "Missing function markers in {}:\n{}\n{}".format(
                label,
                start_marker,
                end_marker,
            )
        )
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    segment = text[start:end]
    if line_marker not in segment:
        raise SystemExit(
            "Missing line marker in {}:\n{}".format(label, line_marker)
        )
    idx = segment.rindex(line_marker)
    return text[: start + idx] + block + text[start + idx :]


def patch_config(config_path):
    text = _read(config_path)
    if "temporal_memory_track_query_head_enabled" in text:
        print("config already patched:", config_path)
        return
    pattern = re.compile(
        r"\nTEMPORAL_MEMORY:\n(.*?)(?=\n[A-Z_]+:\n|\Z)",
        re.S,
    )
    match = pattern.search(text)
    if match is None:
        raise SystemExit("Could not locate TEMPORAL_MEMORY: section.")
    text = text[: match.end(1)] + CONFIG_SNIPPET + text[match.end(1) :]
    _write(config_path, text)
    print("patched config:", config_path)


def patch_model(path):
    text = _read(path)
    if "TrackQueryHead" in text:
        print("model already patched:", path)
        return

    import_candidates = [
        "from model.temporal_frame_net import TemporalFrameNet\n",
        "class TemporalSelfAttentionMemory(nn.Module):\n",
        "class ConvGRUCell(nn.Module):\n",
        "class BidirectionalTemporalMemoryNet(nn.Module):\n",
    ]
    patched_import = False
    for import_marker in import_candidates:
        if import_marker in text:
            text = _insert_before_required(
                text,
                import_marker,
                "from model.object_track_head import TrackQueryHead\n",
                path,
            )
            patched_import = True
            break
    if not patched_import:
        raise SystemExit(
            "Could not find a suitable import marker in {}.".format(path)
        )

    signature_marker = "        density_calibration_enabled=False,\n"
    text = _insert_before_required(
        text,
        signature_marker,
        "        track_query_head_enabled=False,\n"
        "        track_query_num_queries=32,\n"
        "        track_query_hidden=128,\n"
        "        track_query_max_flow=2.0,\n",
        path,
    )

    init_marker = "    @property\n    def input_channels(self):\n"
    text = _insert_before_required(
        text,
        init_marker,
        "        self.track_query_head_enabled = bool(track_query_head_enabled)\n"
        "        self.track_query_head = None\n"
        "        self._last_track_query_outputs = None\n"
        "        if self.track_query_head_enabled:\n"
        "            self.track_query_head = TrackQueryHead(\n"
        "                channels=bottleneck_channels,\n"
        "                num_queries=int(track_query_num_queries),\n"
        "                hidden=int(track_query_hidden),\n"
        "                max_flow=float(track_query_max_flow),\n"
        "            )\n",
        path,
    )

    residual_block = (
        "        if (\n"
        "            getattr(self, 'track_query_head_enabled', False)\n"
        "            and self.track_query_head is not None\n"
        "        ):\n"
        "            track_outputs = self.track_query_head(bottlenecks)\n"
        "            self._last_track_query_outputs = track_outputs\n"
        "            residual = residual + track_outputs['residual']\n"
        "\n"
    )
    text = _insert_before_last_in_function(
        text,
        "    def _memory_residual(self, bottlenecks):\n",
        "    def temporal_residual(self, bottlenecks):\n",
        "        return residual\n",
        residual_block,
        path,
    )
    _write(path, text)
    print("patched model:", path)


def patch_train(path):
    text = _read(path)
    if "track_query_loss" in text:
        print("train already patched:", path)
        return

    import_marker = "from utils.track_aware_loss import track_aware_event_loss\n"
    if import_marker not in text:
        import_marker = "from utils.component_hard_negative import (\n"
    text = _replace_required(
        text,
        import_marker,
        "from utils.object_track_loss import track_query_loss\n",
        path,
    )

    load_sig_marker = "    local_temporal_context_enabled=False,\n"
    text = _insert_before_required(
        text,
        load_sig_marker,
        "    track_query_head_enabled=False,\n",
        path,
    )

    adding_marker = (
        "        adding_local_temporal_context = bool(\n"
        "            local_temporal_context_enabled\n"
        "            and not bool(saved_memory.get('local_temporal_context_enabled', False))\n"
        "        )\n"
    )
    text = _replace_required(
        text,
        adding_marker,
        "        adding_track_query_head = bool(\n"
        "            track_query_head_enabled\n"
        "            and not bool(saved_memory.get('track_query_head_enabled', False))\n"
        "        )\n",
        path,
    )

    strict_marker = "                or adding_local_temporal_context\n            ),\n"
    text = _replace_required(
        text,
        strict_marker,
        "                or adding_track_query_head\n",
        path,
    )

    expected_condition_marker = (
        "            or adding_local_temporal_context\n        ):\n"
    )
    text = _replace_required(
        text,
        expected_condition_marker,
        "            or adding_track_query_head\n",
        path,
    )

    expected_block = (
        "            if adding_track_query_head:\n"
        "                expected_missing.update(\n"
        "                    'track_query_head.' + name\n"
        "                    for name in model.track_query_head.state_dict()\n"
        "                )\n"
    )
    validation_marker = (
        "            if (\n"
        "                set(load_result.missing_keys) != expected_missing\n"
    )
    text = _insert_before_required(
        text,
        validation_marker,
        expected_block,
        path,
    )

    config_block = (
        "    track_query_head_enabled = bool(\n"
        "        getattr(cfg, 'temporal_memory_track_query_head_enabled', False)\n"
        "    )\n"
        "    track_query_num_queries = int(\n"
        "        getattr(cfg, 'temporal_memory_track_query_num_queries', 32)\n"
        "    )\n"
        "    track_query_hidden = int(\n"
        "        getattr(cfg, 'temporal_memory_track_query_hidden', 128)\n"
        "    )\n"
        "    track_query_max_flow = float(\n"
        "        getattr(cfg, 'temporal_memory_track_query_max_flow', 2.0)\n"
        "    )\n"
        "    track_query_loss_weight = float(\n"
        "        getattr(cfg, 'temporal_memory_track_query_loss_weight', 0.05)\n"
        "    )\n"
        "    track_query_warmup_epochs = int(\n"
        "        getattr(cfg, 'temporal_memory_track_query_warmup_epochs', 3)\n"
        "    )\n"
        "    track_query_lr_multiplier = float(\n"
        "        getattr(cfg, 'temporal_memory_track_query_lr_multiplier', 1.0)\n"
        "    )\n"
    )
    model_marker = "    model = BidirectionalTemporalMemoryNet(\n"
    text = _insert_before_required(text, model_marker, config_block + "\n", path)

    model_args_marker = "        center_memory_downsample=center_memory_downsample,\n"
    text = _replace_required(
        text,
        model_args_marker,
        "        track_query_head_enabled=track_query_head_enabled,\n"
        "        track_query_num_queries=track_query_num_queries,\n"
        "        track_query_hidden=track_query_hidden,\n"
        "        track_query_max_flow=track_query_max_flow,\n",
        path,
    )

    load_call_marker = "        local_temporal_context_enabled=local_temporal_context_enabled,\n"
    text = _replace_required(
        text,
        load_call_marker,
        "        track_query_head_enabled=track_query_head_enabled,\n",
        path,
    )

    optimizer_block = (
        "    track_query_multiplier = float(\n"
        "        getattr(config, 'temporal_memory_track_query_lr_multiplier', 1.0)\n"
        "    )\n"
        "    track_query_parameters = []\n"
        "    if getattr(model, 'track_query_head_enabled', False):\n"
        "        track_query_parameters = list(model.track_query_head.parameters())\n"
        "    if track_query_parameters:\n"
        "        parameter_groups.append(\n"
        "            {\n"
        "                'name': 'track_query',\n"
        "                'params': track_query_parameters,\n"
        "                'lr': float(config.lr) * track_query_multiplier,\n"
        "            }\n"
        "        )\n"
    )
    optimizer_marker = (
        "    return optim.AdamW(parameter_groups, weight_decay=1e-4)\n"
    )
    text = _insert_before_required(
        text,
        optimizer_marker,
        optimizer_block,
        path,
    )

    loss_block = (
        "            track_query_loss_value = event_logits.sum() * 0.0\n"
        "            if track_query_head_enabled and epoch >= track_query_warmup_epochs:\n"
        "                track_query_outputs = getattr(\n"
        "                    model, '_last_track_query_outputs', None\n"
        "                )\n"
        "                if track_query_outputs is not None:\n"
        "                    track_query_loss_value, _ = track_query_loss(\n"
        "                        track_query_outputs,\n"
        "                        model.track_query_head.anchors,\n"
        "                        labels,\n"
        "                        target_ids,\n"
        "                        event_time_indices,\n"
        "                        event_x,\n"
        "                        event_y,\n"
        "                        int(cfg.res[1]),\n"
        "                        int(cfg.res[0]),\n"
        "                        model.track_query_head.num_queries,\n"
        "                        frames.shape[1],\n"
        "                    )\n"
        "                    loss = (\n"
        "                        loss\n"
        "                        + track_query_loss_weight * track_query_loss_value\n"
        "                    )\n"
    )
    backward_marker = "            loss.backward()\n"
    text = _insert_before_required(text, backward_marker, loss_block, path)

    sum_init_marker = "        loss_sum = 0.0\n"
    text = _replace_required(
        text,
        sum_init_marker,
        "        track_query_loss_sum = 0.0\n",
        path,
    )
    sum_accum_marker = "            loss_sum += float(loss.detach().item())\n"
    text = _replace_required(
        text,
        sum_accum_marker,
        "            track_query_loss_sum += float(track_query_loss_value.detach().item())\n",
        path,
    )

    metadata_marker = "            'temporal_memory': {\n"
    text = _replace_required(
        text,
        metadata_marker,
        "                'track_query_head_enabled': track_query_head_enabled,\n"
        "                'track_query_num_queries': track_query_num_queries,\n"
        "                'track_query_hidden': track_query_hidden,\n"
        "                'track_query_max_flow': track_query_max_flow,\n",
        path,
    )
    _write(path, text)
    print("patched train:", path)


def patch_inference(path):
    text = _read(path)
    if "saved_track_query_head" in text:
        print("inference already patched:", path)
        return
    saved_marker = (
        "    saved_local_temporal_context_kernel_size = int(\n"
        "        saved.get('local_temporal_context_kernel_size', 11)\n"
        "    )\n"
    )
    text = _replace_required(
        text,
        saved_marker,
        "    saved_track_query_head = bool(\n"
        "        saved.get('track_query_head_enabled', False)\n"
        "    )\n"
        "    saved_track_query_num_queries = int(\n"
        "        saved.get('track_query_num_queries', 32)\n"
        "    )\n"
        "    saved_track_query_hidden = int(\n"
        "        saved.get('track_query_hidden', 128)\n"
        "    )\n"
        "    saved_track_query_max_flow = float(\n"
        "        saved.get('track_query_max_flow', 2.0)\n"
        "    )\n",
        path,
    )
    args_marker = "        center_memory_downsample=saved_center_memory_downsample,\n"
    text = _replace_required(
        text,
        args_marker,
        "        track_query_head_enabled=saved_track_query_head,\n"
        "        track_query_num_queries=saved_track_query_num_queries,\n"
        "        track_query_hidden=saved_track_query_hidden,\n"
        "        track_query_max_flow=saved_track_query_max_flow,\n",
        path,
    )
    _write(path, text)
    print("patched inference:", path)


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    patch_config(repo / "configs" / "evisseg_evuav.yaml")
    patch_model(repo / "model" / "temporal_memory_net.py")
    patch_train(repo / "train_temporal_memory.py")
    patch_inference(repo / "utils" / "temporal_memory_inference.py")
    print("TrackQueryHead patch applied.")


if __name__ == "__main__":
    main()
