"""
Script to generate evaluation plots from a dataset.

Usage:
    uv run --locked utilities/plot_eval.py --dataset=@positronic.cfg.analysis.stacking_episodes --output=eval_plots.html
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np
import plotly.graph_objects as go
import pos3
from plotly.subplots import make_subplots

import positronic.cfg.analysis
from positronic.dataset import Dataset


def success_rate_points(
    data_by_ckpt: dict[str, list[dict[str, Any]]], ckpts: list[str]
) -> tuple[list[str], list[float]]:
    """The checkpoints with a scored episode, and the success rate over their scored episodes, in percent."""
    rated_ckpts, success_rates = [], []
    for ckpt in ckpts:
        scored = [e['success'] for e in data_by_ckpt[ckpt] if e['success'] is not None]
        if scored:
            rated_ckpts.append(ckpt)
            success_rates.append(sum(scored) / len(scored) * 100)
    return rated_ckpts, success_rates


@cfn.config(dataset=positronic.cfg.analysis.stacking_episodes, output='eval_plots.html')
def main(dataset: Dataset, output: str):  # noqa: C901
    """
    Generate evaluation plots for a given dataset.

    Args:
        dataset: The dataset to evaluate. Defaults to stacking_episodes config.
        output: Path to the output HTML file.
    """
    output_path = Path(output).expanduser()
    print(f'Loading dataset with {len(dataset)} episodes...')

    data_by_ckpt = defaultdict(list)

    # Metrics to collect
    # Based on positronic/cfg/analysis.py
    metrics = ['max_stacking_success', 'success_time', 'box_distance_progress', 'movement']

    print('Processing episodes...')
    for i, ep in enumerate(dataset):
        # Accessing keys triggers the lazy transformation
        ckpt_raw = ep.get('checkpoint')
        if not ckpt_raw:
            continue

        ckpt = ckpt_raw.replace('\\', '/').split('/')[-1]

        # Collect metrics
        entry: dict[str, Any] = {'id': i}
        for m in metrics:
            val = ep.get(m)
            if val is not None:
                entry[m] = val

        # ``None`` for an episode that ``counted_outcome`` does not score.
        entry['success'] = ep.get('success')

        data_by_ckpt[ckpt].append(entry)

    if not data_by_ckpt:
        print('No checkpoints found in dataset. Exiting.')
        return

    # Sort checkpoints
    # Try to extract step number for sorting: e.g. "dataset\experiment\1000" -> 1000
    def sort_key(ckpt_name):
        try:
            # Assuming format ...\step or just step
            parts = ckpt_name.replace('\\', '/').split('/')
            return int(parts[-1])
        except ValueError:
            return ckpt_name

    sorted_ckpts = sorted(data_by_ckpt.keys(), key=sort_key)

    print(f'Found {len(sorted_ckpts)} checkpoints.')
    print(f'Checkpoints: {sorted_ckpts}')

    # Create plots
    # Layout:
    # 1. Success Rate (%) (Line + Markers)
    # 2. Max Stacking Success (Box Plot)
    # 3. Success Time (Box Plot, only successful)
    # 4. Box Distance Progress (Box Plot)
    # 5. Movement (Box Plot)

    fig = make_subplots(
        rows=5,
        cols=1,
        subplot_titles=(
            'Success Rate (%)',
            'Max Stacking Success (Distribution)',
            'Success Time (s) (Successful Episodes Only)',
            'Box Distance Progress (%)',
            'Movement',
        ),
        vertical_spacing=0.08,
    )

    # 1. Success Rate
    rated_ckpts, success_rates = success_rate_points(data_by_ckpt, sorted_ckpts)
    fig.add_trace(
        go.Scatter(x=rated_ckpts, y=success_rates, mode='lines+markers', name='Success Rate (%)'), row=1, col=1
    )

    # Helper for box plots
    def add_metric_plot(metric_key, row, name, filter_success=False):
        x_vals = []
        y_vals = []
        # Calculate mean per checkpoint for the line plot
        mean_x = []
        mean_y = []

        for ckpt in sorted_ckpts:
            entries = data_by_ckpt[ckpt]
            # Filter values
            vals = [
                e[metric_key]
                for e in entries
                if metric_key in e and e[metric_key] is not None and (not filter_success or e.get('success'))
            ]

            if vals:
                x_vals.extend([ckpt] * len(vals))
                y_vals.extend(vals)
                mean_x.append(ckpt)
                mean_y.append(np.mean(vals))
            else:
                # If no data for this checkpoint, we might skip it in the mean line
                pass

        # Add Box plot for distribution
        # boxpoints='all' shows all points, jitter spreads them out
        fig.add_trace(
            go.Box(x=x_vals, y=y_vals, name=name, boxpoints='all', jitter=0.3, pointpos=-1.8, showlegend=False),
            row=row,
            col=1,
        )

        # Add Mean line
        fig.add_trace(
            go.Scatter(
                x=mean_x,
                y=mean_y,
                mode='lines+markers',
                name=f'{name} (Mean)',
                line={'width': 2, 'color': 'black'},
                showlegend=False,
            ),
            row=row,
            col=1,
        )

    add_metric_plot('max_stacking_success', 2, 'Max Success')
    add_metric_plot('success_time', 3, 'Time', filter_success=True)
    add_metric_plot('box_distance_progress', 4, 'Box Progress')
    add_metric_plot('movement', 5, 'Movement')

    fig.update_layout(
        height=2000,
        width=1200,
        title_text='Evaluation Metrics by Checkpoint',
        showlegend=False,  # Hiding legend to avoid clutter, titles are enough
    )

    print(f'Saving plots to {output_path}...')
    fig.write_html(output_path)
    print(f'Done. Open {output_path} in your browser to view results.')


if __name__ == '__main__':
    with pos3.mirror():
        cfn.cli(main)
