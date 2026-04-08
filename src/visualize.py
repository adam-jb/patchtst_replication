"""Generate interactive HTML reports from training results."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# Paper reference values by dataset
# ETTh1: Table 3 PatchTST/42 (L=336). No self-supervised in paper for ETTh1.
# Weather: Table 4 (L=512, P=12, S=12 for self-supervised)
PAPER_DATA = {
    "etth1": {
        "supervised": {
            96: {"mse": 0.375, "mae": 0.399},
            192: {"mse": 0.414, "mae": 0.421},
            336: {"mse": 0.431, "mae": 0.436},
            720: {"mse": 0.449, "mae": 0.466},
        },
        "supervised_64": {
            96: {"mse": 0.370, "mae": 0.400},
            192: {"mse": 0.413, "mae": 0.429},
            336: {"mse": 0.422, "mae": 0.440},
            720: {"mse": 0.447, "mae": 0.468},
        },
        "selfsup_ft": {},  # not in paper
        "selfsup_lp": {},  # not in paper
        "table_label": "Table 3, PatchTST/42, L=336",
        "title": "ETTh1",
    },
    "weather": {
        "supervised": {
            96: {"mse": 0.152, "mae": 0.199},
            192: {"mse": 0.197, "mae": 0.243},
            336: {"mse": 0.249, "mae": 0.283},
            720: {"mse": 0.320, "mae": 0.335},
        },
        "selfsup_ft": {
            96: {"mse": 0.144, "mae": 0.193},
            192: {"mse": 0.190, "mae": 0.236},
            336: {"mse": 0.244, "mae": 0.280},
            720: {"mse": 0.320, "mae": 0.335},
        },
        "selfsup_lp": {
            96: {"mse": 0.158, "mae": 0.209},
            192: {"mse": 0.203, "mae": 0.249},
            336: {"mse": 0.251, "mae": 0.285},
            720: {"mse": 0.321, "mae": 0.336},
        },
        "table_label": "Table 4, Weather, L=512",
        "title": "Weather",
    },
}

# Backward compat aliases (ETTh1 defaults)
PAPER_SUPERVISED = PAPER_DATA["etth1"]["supervised"]
PAPER_RESULTS = PAPER_SUPERVISED

HORIZONS = [96, 192, 336, 720]
CHANNEL_NAMES_MAP = {
    "etth1": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
    "weather": [f"Ch{i}" for i in range(21)],  # Weather has 21 unnamed channels
}


def load_json(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def generate_supervised_report(results_dir: str = "results", dataset: str = "etth1") -> None:
    """Generate supervised_results.html with comparison tables, bar charts, scaling plots, training curves."""

    results = {}
    for T in HORIZONS:
        r = load_json(os.path.join(results_dir, f"{dataset}_supervised_T{T}.json"))
        if r:
            results[T] = r

    if not results:
        print(f"No supervised results found for {dataset}.")
        return

    # Create multi-panel figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "MSE: Ours vs Paper", "MAE: Ours vs Paper",
            "MSE Scaling with Horizon", "MAE Scaling with Horizon",
            "Training Loss Curves", "Validation Loss Curves"
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    horizons_found = sorted(results.keys())
    paper_ref = PAPER_DATA.get(dataset, PAPER_DATA["etth1"])["supervised"]
    ds_title = PAPER_DATA.get(dataset, PAPER_DATA["etth1"])["title"]
    table_label = PAPER_DATA.get(dataset, PAPER_DATA["etth1"])["table_label"]

    # Row 1: Grouped bar charts
    ours_mse = [results[T]["test_metrics"]["mse"] for T in horizons_found]
    paper_mse = [paper_ref.get(T, {}).get("mse", 0) for T in horizons_found]
    ours_mae = [results[T]["test_metrics"]["mae"] for T in horizons_found]
    paper_mae = [paper_ref.get(T, {}).get("mae", 0) for T in horizons_found]
    x_labels = [str(T) for T in horizons_found]

    fig.add_trace(go.Bar(name="Ours (MSE)", x=x_labels, y=ours_mse,
                         marker_color="#636EFA", text=[f"{v:.4f}" for v in ours_mse],
                         textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(name="Paper (MSE)", x=x_labels, y=paper_mse,
                         marker_color="#EF553B", text=[f"{v:.4f}" for v in paper_mse],
                         textposition="outside"), row=1, col=1)

    fig.add_trace(go.Bar(name="Ours (MAE)", x=x_labels, y=ours_mae,
                         marker_color="#636EFA", text=[f"{v:.4f}" for v in ours_mae],
                         textposition="outside", showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(name="Paper (MAE)", x=x_labels, y=paper_mae,
                         marker_color="#EF553B", text=[f"{v:.4f}" for v in paper_mae],
                         textposition="outside", showlegend=False), row=1, col=2)

    # Row 2: Scaling line plots
    fig.add_trace(go.Scatter(name="Ours", x=horizons_found, y=ours_mse,
                             mode="lines+markers", line=dict(color="#636EFA", width=3),
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(name="Paper", x=list(paper_ref.keys()),
                             y=[v["mse"] for v in paper_ref.values()],
                             mode="lines+markers", line=dict(color="#EF553B", width=3, dash="dash"),
                             showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(name="Ours", x=horizons_found, y=ours_mae,
                             mode="lines+markers", line=dict(color="#636EFA", width=3),
                             showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(name="Paper", x=list(paper_ref.keys()),
                             y=[v["mae"] for v in paper_ref.values()],
                             mode="lines+markers", line=dict(color="#EF553B", width=3, dash="dash"),
                             showlegend=False), row=2, col=2)

    # Row 3: Training curves
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    for i, T in enumerate(horizons_found):
        history = results[T]["train_history"]
        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]

        fig.add_trace(go.Scatter(name=f"T={T}", x=epochs, y=train_losses,
                                 mode="lines", line=dict(color=colors[i % len(colors)]),
                                 showlegend=(i == 0 or True)), row=3, col=1)
        fig.add_trace(go.Scatter(name=f"T={T}", x=epochs, y=val_losses,
                                 mode="lines", line=dict(color=colors[i % len(colors)]),
                                 showlegend=False), row=3, col=2)

    # Axis labels
    for col in [1, 2]:
        fig.update_xaxes(title_text="Prediction Horizon", row=1, col=col)
        fig.update_xaxes(title_text="Prediction Horizon", row=2, col=col)
        fig.update_xaxes(title_text="Epoch", row=3, col=col)
    fig.update_yaxes(title_text="MSE", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=1, col=2)
    fig.update_yaxes(title_text="MSE", row=2, col=1)
    fig.update_yaxes(title_text="MAE", row=2, col=2)
    fig.update_yaxes(title_text="Loss", row=3, col=1)
    fig.update_yaxes(title_text="Loss", row=3, col=2)

    fig.update_layout(
        title=f"PatchTST Supervised Results — {ds_title}",
        height=1200, width=1100,
        barmode="group",
        template="plotly_white",
    )

    # Build comparison table HTML
    table_html = _build_comparison_table(results, "Supervised", dataset)

    # Write HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>PatchTST Supervised Results — {ds_title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #fafafa; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px 16px; text-align: center; }}
        th {{ background: #f0f0f0; font-weight: 600; }}
        .good {{ background: #d4edda; }}
        .ok {{ background: #fff3cd; }}
        .bad {{ background: #f8d7da; }}
    </style>
</head>
<body>
<h1>PatchTST Supervised Results — {ds_title}</h1>
{table_html}
{fig.to_html(full_html=False, include_plotlyjs=True)}
</body>
</html>"""

    out_path = os.path.join(results_dir, f"{dataset}_supervised_results.html")
    with open(out_path, "w") as f:
        f.write(html_content)
    print(f"Supervised report: {out_path}")


def generate_forecast_report(results_dir: str = "results", dataset: str = "etth1") -> None:
    """Generate forecasts.html with sample prediction plots."""

    results = {}
    for T in HORIZONS:
        r = load_json(os.path.join(results_dir, f"{dataset}_supervised_T{T}.json"))
        if r and "sample_predictions" in r:
            results[T] = r

    if not results:
        print(f"No forecast samples found for {dataset}.")
        return

    horizons_found = sorted(results.keys())
    n_samples = min(3, min(len(results[T]["sample_predictions"]) for T in horizons_found))
    # Show OT channel (index 6) and HUFL (index 0)
    channels_to_show = [(6, "OT"), (0, "HUFL")]

    figs_html = []
    for T in horizons_found:
        r = results[T]
        for sample_idx in range(n_samples):
            for ch_idx, ch_name in channels_to_show:
                fig = go.Figure()
                inp = np.array(r["sample_inputs"][sample_idx][ch_idx])
                tgt = np.array(r["sample_targets"][sample_idx][ch_idx])
                pred = np.array(r["sample_predictions"][sample_idx][ch_idx])

                seq_len = len(inp)
                pred_len = len(tgt)

                # Lookback
                fig.add_trace(go.Scatter(
                    x=list(range(seq_len)), y=inp.tolist(),
                    mode="lines", name="Lookback", line=dict(color="gray", width=1.5)
                ))
                # Ground truth
                fig.add_trace(go.Scatter(
                    x=list(range(seq_len, seq_len + pred_len)), y=tgt.tolist(),
                    mode="lines", name="Ground Truth", line=dict(color="#636EFA", width=2)
                ))
                # Prediction
                fig.add_trace(go.Scatter(
                    x=list(range(seq_len, seq_len + pred_len)), y=pred.tolist(),
                    mode="lines", name="Prediction", line=dict(color="#EF553B", width=2, dash="dash")
                ))

                fig.update_layout(
                    title=f"T={T} | {ch_name} | Sample {sample_idx+1}",
                    xaxis_title="Timestep", yaxis_title="Normalised Value",
                    height=300, width=900, margin=dict(t=40, b=30),
                    template="plotly_white", showlegend=True,
                )
                figs_html.append(fig.to_html(full_html=False, include_plotlyjs=False))

    ds_title = PAPER_DATA.get(dataset, PAPER_DATA["etth1"])["title"]
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>PatchTST Forecast Samples — {ds_title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #fafafa; }}
        h1 {{ color: #333; }}
    </style>
</head>
<body>
<h1>PatchTST Forecast Samples — {ds_title}</h1>
<p>Showing lookback context (gray), ground truth (blue), and model prediction (red dashed).</p>
{''.join(figs_html)}
</body>
</html>"""

    out_path = os.path.join(results_dir, f"{dataset}_forecasts.html")
    with open(out_path, "w") as f:
        f.write(html_content)
    print(f"Forecast report: {out_path}")


def generate_selfsup_report(results_dir: str = "results", dataset: str = "etth1") -> None:
    """Generate selfsup_results.html comparing supervised, linear probe, and fine-tuned."""

    # Load all results
    sup_results = {}
    probe_results = {}
    ft_results = {}
    for T in HORIZONS:
        r = load_json(os.path.join(results_dir, f"{dataset}_supervised_T{T}.json"))
        if r:
            sup_results[T] = r
        r = load_json(os.path.join(results_dir, f"{dataset}_selfsup_linear_probe_T{T}.json"))
        if r:
            probe_results[T] = r
        r = load_json(os.path.join(results_dir, f"{dataset}_selfsup_finetune_T{T}.json"))
        if r:
            ft_results[T] = r

    pretrain_hist = load_json(os.path.join(results_dir, f"{dataset}_pretrain_history.json"))

    if not (probe_results or ft_results):
        print("No self-supervised results found.")
        return

    horizons_found = sorted(set(list(probe_results.keys()) + list(ft_results.keys())))

    # Build comparison figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "MSE by Method", "MAE by Method",
            "Pretraining Loss", "Fine-tuning Loss Curves"
        ],
        vertical_spacing=0.15,
    )

    paper_ref = PAPER_DATA.get(dataset, PAPER_DATA["etth1"])
    ds_title = paper_ref["title"]
    table_label = paper_ref["table_label"]

    # Paper reference + our results
    methods = [
        (f"Paper Supervised ({table_label})", paper_ref["supervised"], "#888888", "dash"),
    ]
    if paper_ref["selfsup_lp"]:
        methods.append((f"Paper Lin. Probe ({table_label})", paper_ref["selfsup_lp"], "#AAAAAA", "dot"))
    if paper_ref["selfsup_ft"]:
        methods.append((f"Paper Fine-tune ({table_label})", paper_ref["selfsup_ft"], "#CCCCCC", "dashdot"))
    methods += [
        ("Our Supervised", sup_results, "#636EFA", "solid"),
        ("Our Linear Probe", probe_results, "#00CC96", "solid"),
        ("Our Fine-tuned", ft_results, "#EF553B", "solid"),
    ]

    for name, data, color, dash in methods:
        if not data:
            continue
        if isinstance(data, dict) and all(isinstance(v, dict) and "mse" in v for v in data.values()):
            # Paper reference data
            mse_vals = [data.get(T, {}).get("mse") for T in horizons_found]
            mae_vals = [data.get(T, {}).get("mae") for T in horizons_found]
        else:
            mse_vals = [data[T]["test_metrics"]["mse"] if T in data else None for T in horizons_found]
            mae_vals = [data[T]["test_metrics"]["mae"] if T in data else None for T in horizons_found]

        fig.add_trace(go.Scatter(
            name=name, x=horizons_found, y=mse_vals,
            mode="lines+markers", line=dict(color=color, width=2.5, dash=dash),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            name=name, x=horizons_found, y=mae_vals,
            mode="lines+markers", line=dict(color=color, width=2.5, dash=dash),
            showlegend=False,
        ), row=1, col=2)

    # Pretraining loss
    if pretrain_hist:
        epochs = [h["epoch"] for h in pretrain_hist["pretrain_history"]]
        train_l = [h["train_loss"] for h in pretrain_hist["pretrain_history"]]
        val_l = [h["val_loss"] for h in pretrain_hist["pretrain_history"]]
        fig.add_trace(go.Scatter(name="Train", x=epochs, y=train_l,
                                 mode="lines", line=dict(color="#636EFA")), row=2, col=1)
        fig.add_trace(go.Scatter(name="Val", x=epochs, y=val_l,
                                 mode="lines", line=dict(color="#EF553B")), row=2, col=1)

    # Fine-tuning curves
    colors_ft = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    for i, T in enumerate(horizons_found):
        if T in ft_results:
            hist = ft_results[T]["train_history"]
            epochs_ft = [h["epoch"] for h in hist]
            val_l = [h["val_loss"] for h in hist]
            fig.add_trace(go.Scatter(
                name=f"T={T}", x=epochs_ft, y=val_l,
                mode="lines", line=dict(color=colors_ft[i % len(colors_ft)]),
                showlegend=False,
            ), row=2, col=2)

    fig.update_xaxes(title_text="Horizon", row=1, col=1)
    fig.update_xaxes(title_text="Horizon", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    fig.update_yaxes(title_text="MSE", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_yaxes(title_text="Val Loss", row=2, col=2)

    fig.update_layout(
        title=f"PatchTST Self-Supervised Results — {ds_title}",
        height=900, width=1100,
        template="plotly_white",
    )

    # Comparison table with paper values from the correct table
    p_sup_ref = paper_ref["supervised"]
    p_lp_ref = paper_ref["selfsup_lp"]
    p_ft_ref = paper_ref["selfsup_ft"]
    has_paper_selfsup = bool(p_lp_ref)

    def _fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    table_rows = ""
    for T in horizons_found:
        p_sup = p_sup_ref.get(T, {})
        p_lp = p_lp_ref.get(T, {})
        p_ft = p_ft_ref.get(T, {})
        sup = sup_results.get(T, {}).get("test_metrics", {})
        probe = probe_results.get(T, {}).get("test_metrics", {})
        ft = ft_results.get(T, {}).get("test_metrics", {})

        table_rows += f"""<tr>
            <td>{T}</td>
            <td>{_fmt(p_sup.get('mse', '-'))}</td><td>{_fmt(p_sup.get('mae', '-'))}</td>"""
        if has_paper_selfsup:
            table_rows += f"""
            <td>{_fmt(p_lp.get('mse', '-'))}</td><td>{_fmt(p_lp.get('mae', '-'))}</td>
            <td>{_fmt(p_ft.get('mse', '-'))}</td><td>{_fmt(p_ft.get('mae', '-'))}</td>"""
        table_rows += f"""
            <td>{_fmt(sup.get('mse', '-'))}</td><td>{_fmt(sup.get('mae', '-'))}</td>
            <td>{_fmt(probe.get('mse', '-'))}</td><td>{_fmt(probe.get('mae', '-'))}</td>
            <td>{_fmt(ft.get('mse', '-'))}</td><td>{_fmt(ft.get('mae', '-'))}</td>
        </tr>"""

    paper_lp_ft_headers = ""
    if has_paper_selfsup:
        paper_lp_ft_headers = """
            <th colspan="2">Paper Lin. Probe</th>
            <th colspan="2">Paper Fine-tune</th>"""
    paper_lp_ft_subheaders = ""
    if has_paper_selfsup:
        paper_lp_ft_subheaders = """
            <th>MSE</th><th>MAE</th>
            <th>MSE</th><th>MAE</th>"""

    note_html = ""
    if not has_paper_selfsup:
        note_html = """<p><strong>Note:</strong> The paper (Table 4) does NOT include self-supervised results for ETTh1 —
        only Weather, Traffic, and Electricity. Our self-supervised ETTh1 experiments are <strong>novel work</strong>.</p>"""

    table_html = f"""
    <table>
        <tr>
            <th rowspan="2">Horizon</th>
            <th colspan="2">Paper Supervised</th>{paper_lp_ft_headers}
            <th colspan="2">Our Supervised</th>
            <th colspan="2">Our Linear Probe</th>
            <th colspan="2">Our Fine-tuned</th>
        </tr>
        <tr>
            <th>MSE</th><th>MAE</th>{paper_lp_ft_subheaders}
            <th>MSE</th><th>MAE</th>
            <th>MSE</th><th>MAE</th>
            <th>MSE</th><th>MAE</th>
        </tr>
        {table_rows}
    </table>
    {note_html}
    <p><em>Paper values from {table_label} of
    <a href="https://arxiv.org/abs/2211.14730">Nie et al. (2023) "A Time Series is Worth 64 Words"</a>.</em></p>"""

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>PatchTST Self-Supervised Results — {ds_title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #fafafa; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px 16px; text-align: center; }}
        th {{ background: #f0f0f0; font-weight: 600; }}
    </style>
</head>
<body>
<h1>PatchTST Self-Supervised Results — {ds_title}</h1>
{table_html}
{fig.to_html(full_html=False, include_plotlyjs=True)}
</body>
</html>"""

    out_path = os.path.join(results_dir, f"{dataset}_selfsup_results.html")
    with open(out_path, "w") as f:
        f.write(html_content)
    print(f"Self-supervised report: {out_path}")


def _build_comparison_table(results: dict, mode: str, dataset: str = "etth1") -> str:
    """Build an HTML comparison table with color-coded % differences."""
    paper_ref = PAPER_DATA.get(dataset, PAPER_DATA["etth1"])
    paper_sup = paper_ref["supervised"]
    table_label = paper_ref["table_label"]
    ds_title = paper_ref["title"]

    rows = ""
    for T in sorted(results.keys()):
        r = results[T]
        our_mse = r["test_metrics"]["mse"]
        our_mae = r["test_metrics"]["mae"]
        p = paper_sup.get(T, {})
        paper_mse = p.get("mse")
        paper_mae = p.get("mae")

        if paper_mse and paper_mae:
            mse_diff = (our_mse - paper_mse) / paper_mse * 100
            mae_diff = (our_mae - paper_mae) / paper_mae * 100

            def css_class(diff):
                if abs(diff) <= 5:
                    return "good"
                elif abs(diff) <= 10:
                    return "ok"
                return "bad"

            rows += f"""<tr>
                <td><strong>{T}</strong></td>
                <td>{our_mse:.4f}</td><td>{paper_mse:.3f}</td>
                <td class="{css_class(mse_diff)}">{mse_diff:+.1f}%</td>
                <td>{our_mae:.4f}</td><td>{paper_mae:.3f}</td>
                <td class="{css_class(mae_diff)}">{mae_diff:+.1f}%</td>
            </tr>"""
        else:
            rows += f"""<tr>
                <td><strong>{T}</strong></td>
                <td>{our_mse:.4f}</td><td>-</td><td>-</td>
                <td>{our_mae:.4f}</td><td>-</td><td>-</td>
            </tr>"""

    # Training time
    total_time = sum(
        sum(h["time_s"] for h in results[T]["train_history"])
        for T in results
    )
    total_epochs = sum(len(results[T]["train_history"]) for T in results)

    return f"""
    <h2>{mode} — Comparison with Paper ({table_label})</h2>
    <table>
        <tr>
            <th>Horizon</th>
            <th>Our MSE</th><th>Paper MSE</th><th>MSE Diff</th>
            <th>Our MAE</th><th>Paper MAE</th><th>MAE Diff</th>
        </tr>
        {rows}
    </table>
    <p><em>Color: green = within 5%, yellow = 5-10%, red = >10%. Paper numbers from {table_label} of
    <a href="https://arxiv.org/abs/2211.14730">Nie et al. (2023) "A Time Series is Worth 64 Words"</a>,
    {ds_title}.</em></p>
    <p>Total training: {total_epochs} epochs in {total_time:.0f}s ({total_time/60:.1f} min)</p>
    """


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="etth1", choices=["etth1", "weather"])
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "supervised", "forecasts", "selfsup"])
    args = parser.parse_args()

    if args.mode in ("all", "supervised"):
        generate_supervised_report(args.results_dir, args.dataset)
    if args.mode in ("all", "forecasts"):
        generate_forecast_report(args.results_dir, args.dataset)
    if args.mode in ("all", "selfsup"):
        generate_selfsup_report(args.results_dir, args.dataset)
