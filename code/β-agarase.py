import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def plot_dumbbell(
    df,
    save_path,
    id_col="uniprot_id",
    true_col="topt",
    pred_col="pred_topt"
):

    required_cols = [id_col, true_col, pred_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"The input file is missing necessary columns: {missing_cols}")

    plot_df = df[required_cols].copy()
    plot_df[id_col] = plot_df[id_col].astype(str)
    plot_df[true_col] = pd.to_numeric(plot_df[true_col], errors="coerce")
    plot_df[pred_col] = pd.to_numeric(plot_df[pred_col], errors="coerce")
    plot_df = plot_df.dropna(subset=required_cols).reset_index(drop=True)

    if len(plot_df) == 0:
        raise ValueError("There is no data available for drawing after cleaning.")

    y_true = plot_df[true_col].to_numpy()
    y_pred = plot_df[pred_col].to_numpy()
    y_pos = np.arange(len(plot_df))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    experimental_color = "#0072B2"
    predicted_color = "#D55E00"
    connector_color = "#A6A6A6"
    edge_color = "#202020"
    grid_color = "#D9D9D9"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.edgecolor": edge_color,
        "axes.linewidth": 1.4,
        "xtick.color": edge_color,
        "ytick.color": edge_color,
        "text.color": edge_color
    })

    fig_height = max(8.5, len(plot_df) * 0.28)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))

    for i, (true_value, pred_value) in enumerate(zip(y_true, y_pred)):
        ax.plot(
            [true_value, pred_value],
            [i, i],
            color=connector_color,
            linewidth=1.6,
            alpha=0.9,
            zorder=1
        )

    ax.scatter(
        y_true,
        y_pos,
        s=78,
        color=experimental_color,
        edgecolor=edge_color,
        linewidth=0.8,
        label="Experimental Topt",
        zorder=3
    )

    ax.scatter(
        y_pred,
        y_pos,
        s=78,
        color=predicted_color,
        edgecolor=edge_color,
        linewidth=0.8,
        label="Predicted Topt",
        zorder=3
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        plot_df[id_col],
        fontsize=9.5,
        fontweight="bold"
    )
    ax.invert_yaxis()

    x_min = min(y_true.min(), y_pred.min()) - 3
    x_max = max(y_true.max(), y_pred.max()) + 3
    ax.set_xlim(x_min, x_max)

    ax.set_xlabel(
        "Topt (°C)",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_ylabel(
        "β-Agarase",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_title(
        "Experimental and Predicted Topt of 32 β-Agarases",
        fontsize=16,
        fontweight="bold",
        pad=14
    )

    ax.grid(
        axis="x",
        linestyle=":",
        linewidth=1.0,
        color=grid_color,
        alpha=0.8
    )

    ax.text(
        0.03,
        0.97,
        f"RMSE = {rmse:.2f} °C\nMAE = {mae:.2f} °C",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor=edge_color,
            linewidth=1.2,
            alpha=0.96
        )
    )

    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=11
    )
    legend.get_frame().set_edgecolor(edge_color)
    legend.get_frame().set_linewidth(1.1)
    for txt in legend.get_texts():
        txt.set_fontweight("bold")

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(10.5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color(edge_color)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    svg_path = os.path.splitext(save_path)[0] + ".svg"
    plt.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"PNG saved: {save_path}")
    print(f"SVG saved: {svg_path}")
    print(f"RMSE = {rmse:.2f} °C")
    print(f"MAE = {mae:.2f} °C")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="/home/ys/new/TGC-Net/casestudy/casestudy1/32-result.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/ys/new/TGC-Net/casestudy/casestudy1/32.png"
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default="uniprot_id"
    )
    parser.add_argument(
        "--true_col",
        type=str,
        default="topt"
    )
    parser.add_argument(
        "--pred_col",
        type=str,
        default="pred_topt"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    plot_dumbbell(
        df=df,
        save_path=args.output,
        id_col=args.id_col,
        true_col=args.true_col,
        pred_col=args.pred_col
    )


if __name__ == "__main__":
    main()
