"""Object-level track query head for the temporal-memory backbone.

This head treats tiny UAVs as track queries instead of per-event pixels:

1. bottleneck frames are pooled into a small spatial grid;
2. a fixed set of learned queries cross-attends to the full temporal grid;
3. each query predicts objectness, a constant-velocity center model,
   and a scale;
4. a zero-initialized projection converts the predicted center heatmaps
   into a bottleneck residual, so attaching it to M26 initially changes
   nothing.

The head is intentionally self-contained: it can be trained with a
lightweight Hungarian-style anchor assignment and evaluated with the same
event-level scoring path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackQueryHead(nn.Module):
    def __init__(
        self,
        channels,
        num_queries=32,
        hidden=128,
        num_heads=4,
        grid_h=8,
        grid_w=8,
        max_flow=2.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_queries = int(num_queries)
        self.hidden = int(hidden)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.max_flow = float(max_flow)

        # Deterministic normalized anchors (K over an HxW grid).
        ys, xs = torch.meshgrid(
            torch.linspace(0.0, 1.0, self.grid_h),
            torch.linspace(0.0, 1.0, self.grid_w),
            indexing="ij",
        )
        anchors = torch.stack((xs.reshape(-1), ys.reshape(-1)), dim=1)
        if anchors.shape[0] < self.num_queries:
            raise ValueError("num_queries exceeds grid anchor count.")
        anchors = anchors[: self.num_queries]
        self.register_buffer("anchors", anchors)

        self.query_embed = nn.Parameter(
            torch.randn(self.num_queries, self.hidden) * 0.02
        )
        self.token_proj = nn.Linear(self.channels, self.hidden)
        self.cross_attn = nn.MultiheadAttention(
            self.hidden,
            int(num_heads),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.hidden)
        self.objectness = nn.Linear(self.hidden, 1)
        self.center_offset = nn.Linear(self.hidden, 2)
        self.velocity = nn.Linear(self.hidden, 2)
        self.scale = nn.Linear(self.hidden, 1)

        # Zero-initialized residual: M26 outputs are unchanged at attach time.
        self.heatmap_proj = nn.Conv2d(self.channels, self.channels, 1)
        nn.init.zeros_(self.heatmap_proj.weight)
        nn.init.zeros_(self.heatmap_proj.bias)

    def forward(self, bottlenecks):
        """Return track predictions and a zero-init bottleneck residual."""
        if bottlenecks.ndim != 5:
            raise ValueError("bottlenecks must have shape [B, T, C, H, W].")
        batch_size, sequence_length, channels, height, width = bottlenecks.shape
        if channels != self.channels:
            raise ValueError(
                "TrackQueryHead channels {} != bottleneck channels {}.".format(
                    self.channels,
                    channels,
                )
            )

        pooled = F.adaptive_avg_pool2d(
            bottlenecks.reshape(
                batch_size * sequence_length,
                channels,
                height,
                width,
            ),
            (self.grid_h, self.grid_w),
        ).reshape(
            batch_size,
            sequence_length,
            channels,
            self.grid_h,
            self.grid_w,
        )
        tokens = pooled.permute(0, 1, 3, 4, 2).reshape(
            batch_size,
            sequence_length * self.grid_h * self.grid_w,
            channels,
        )
        tokens = self.token_proj(tokens)

        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        attended = self.norm(queries + attended)

        objectness = self.objectness(attended).squeeze(-1)
        center_offset = torch.tanh(self.center_offset(attended)) * self.max_flow
        velocity = torch.tanh(self.velocity(attended)) * self.max_flow
        scale = torch.sigmoid(self.scale(attended))

        time_norm = (
            torch.arange(sequence_length, device=bottlenecks.device, dtype=torch.float32)
            / max(sequence_length - 1, 1)
        )
        anchors_4d = self.anchors.view(1, self.num_queries, 1, 2)
        centers = (
            anchors_4d
            + center_offset.unsqueeze(2)
            + velocity.unsqueeze(2) * time_norm.view(1, 1, sequence_length, 1)
        ).clamp(0.0, 1.0)

        objectness_prob = torch.sigmoid(objectness)
        support = torch.zeros(
            batch_size,
            sequence_length,
            1,
            height,
            width,
            device=bottlenecks.device,
            dtype=bottlenecks.dtype,
        )
        x_grid = (
            torch.arange(width, device=bottlenecks.device, dtype=torch.float32)
            / max(width - 1, 1)
        )
        y_grid = (
            torch.arange(height, device=bottlenecks.device, dtype=torch.float32)
            / max(height - 1, 1)
        )
        for query_index in range(self.num_queries):
            sigma = (
                scale[:, query_index, 0].clamp(min=0.05, max=0.5)
                * max(height, width)
                * 0.25
            )
            center_x = centers[:, query_index, :, 0]
            center_y = centers[:, query_index, :, 1]
            dx = x_grid.view(1, 1, 1, width) - center_x.view(
                batch_size,
                sequence_length,
                1,
                1,
            )
            dy = y_grid.view(1, 1, height, 1) - center_y.view(
                batch_size,
                sequence_length,
                1,
                1,
            )
            weight = objectness_prob[:, query_index].view(
                batch_size,
                1,
                1,
                1,
            )
            heat = torch.exp(
                -0.5
                * (dx * dx + dy * dy)
                / (sigma.view(batch_size, 1, 1, 1) ** 2 + 1e-6)
            )
            support = support + weight * heat.unsqueeze(2)

        flat_bottleneck = bottlenecks.reshape(
            batch_size * sequence_length,
            channels,
            height,
            width,
        )
        residual = self.heatmap_proj(flat_bottleneck).reshape(
            batch_size,
            sequence_length,
            channels,
            height,
            width,
        )
        residual = residual * support

        return {
            "objectness": objectness,
            "centers": centers,
            "velocity": velocity,
            "scale": scale,
            "support": support,
            "residual": residual,
        }
