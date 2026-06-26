from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required. Install it in your Python environment, then rerun:\n"
        "  pip install matplotlib pandas\n"
        "  python tools\\plot_bspc_figures.py"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs" / "bspc_submission" / "figure_data"
OUT_DIR = ROOT / "docs" / "bspc_submission" / "plots"

DATASET_ORDER = ["BNCI2014001", "BNCI2014002", "BNCI2014004", "Macro average"]
MODEL_ORDER = ["EEGNet", "ShallowConvNet", "DBConformer", "FastGatedConformer"]
ABLATION_ORDER = ["Full", "No channel attention", "Temporal-only"]
COLORS = {
    "EEGNet": "#4C78A8",
    "ShallowConvNet": "#72B7B2",
    "DBConformer": "#F58518",
    "FastGatedConformer": "#54A24B",
    "Full": "#54A24B",
    "No channel attention": "#B279A2",
    "Temporal-only": "#E45756",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
    })


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)


def grouped_bar(ax, data, x_col, group_col, y_col, yerr_col=None, groups=None, x_order=None, colors=None):
    groups = groups or list(data[group_col].drop_duplicates())
    x_order = x_order or list(data[x_col].drop_duplicates())
    x_pos = list(range(len(x_order)))
    width = min(0.8 / max(len(groups), 1), 0.22)

    for idx, group in enumerate(groups):
        subset = data[data[group_col] == group].set_index(x_col)
        values = [subset.loc[x, y_col] if x in subset.index else float("nan") for x in x_order]
        yerr = None
        if yerr_col:
            yerr = [subset.loc[x, yerr_col] if x in subset.index else 0 for x in x_order]
        offsets = [p + (idx - (len(groups) - 1) / 2) * width for p in x_pos]
        ax.bar(
            offsets,
            values,
            width=width,
            label=group,
            color=(colors or COLORS).get(group, None),
            yerr=yerr,
            capsize=3 if yerr_col else 0,
            linewidth=0.5,
            edgecolor="#334155",
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_order, rotation=0)


def add_value_labels(ax, decimals=1, dy=0.5):
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + dy,
            f"{height:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90 if len(ax.patches) > 12 else 0,
        )


def plot_accuracy():
    df = pd.read_csv(DATA_DIR / "main_accuracy.csv")
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df = df.sort_values(["dataset", "model"])

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    grouped_bar(
        ax,
        df,
        x_col="dataset",
        group_col="model",
        y_col="mean_accuracy",
        yerr_col="seed_std",
        groups=MODEL_ORDER,
        x_order=DATASET_ORDER,
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("")
    ax.set_title("Accuracy comparison under LOSO evaluation")
    ax.set_ylim(55, 82)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    add_value_labels(ax, decimals=1, dy=0.35)
    save(fig, "figure_accuracy_comparison")


def plot_parameters():
    df = pd.read_csv(DATA_DIR / "parameter_count.csv")
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER[:-1], ordered=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df = df.sort_values(["dataset", "model"])

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    grouped_bar(
        ax,
        df,
        x_col="dataset",
        group_col="model",
        y_col="parameters",
        groups=MODEL_ORDER,
        x_order=DATASET_ORDER[:-1],
    )
    ax.set_ylabel("Trainable parameters")
    ax.set_xlabel("")
    ax.set_title("Parameter count comparison")
    max_params = df["parameters"].max()
    ax.set_ylim(0, max_params * 1.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height * 1.08,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    save(fig, "figure_parameter_count")


def plot_runtime():
    df = pd.read_csv(DATA_DIR / "runtime_minutes.csv")
    models = ["DBConformer", "FastGatedConformer"]
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER[:-1], ordered=True)
    df["model"] = pd.Categorical(df["model"], models, ordered=True)
    df = df.sort_values(["dataset", "model"])

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    grouped_bar(
        ax,
        df,
        x_col="dataset",
        group_col="model",
        y_col="runtime_minutes",
        groups=models,
        x_order=DATASET_ORDER[:-1],
    )
    ax.set_ylabel("Runtime (min)")
    ax.set_xlabel("")
    ax.set_title("Runtime comparison for five-seed LOSO experiments")
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    add_value_labels(ax, decimals=1, dy=2.0)
    save(fig, "figure_runtime_comparison")


def plot_ablation():
    df = pd.read_csv(DATA_DIR / "ablation_accuracy.csv")
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    df["variant"] = pd.Categorical(df["variant"], ABLATION_ORDER, ordered=True)
    df = df.sort_values(["dataset", "variant"])

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    grouped_bar(
        ax,
        df,
        x_col="dataset",
        group_col="variant",
        y_col="mean_accuracy",
        yerr_col="seed_std",
        groups=ABLATION_ORDER,
        x_order=DATASET_ORDER,
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("")
    ax.set_title("Ablation study of FastGatedConformer")
    ax.set_ylim(68, 79)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    add_value_labels(ax, decimals=1, dy=0.18)
    save(fig, "figure_ablation_study")


def plot_subject_level():
    df = pd.read_csv(DATA_DIR / "subject_level_accuracy.csv")
    models = ["DBConformer", "FastGatedConformer"]
    df = df[df["model"].isin(models)].copy()

    for dataset, sub in df.groupby("dataset"):
        subjects = list(sub["subject"].drop_duplicates())
        fig, ax = plt.subplots(figsize=(9.0, 4.4))
        grouped_bar(
            ax,
            sub,
            x_col="subject",
            group_col="model",
            y_col="mean_accuracy",
            groups=models,
            x_order=subjects,
        )
        ax.set_ylabel("Subject mean accuracy (%)")
        ax.set_xlabel("")
        ax.set_title(f"Subject-level comparison on {dataset}")
        ax.set_ylim(45, 100)
        ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
        save(fig, f"figure_subject_level_{dataset}")


def main():
    setup_style()
    plot_accuracy()
    plot_parameters()
    plot_runtime()
    plot_ablation()
    plot_subject_level()
    print(f"Saved figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
