import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from summarize_logs import parse_log


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "docs" / "bspc_submission" / "figure_data"
DATASETS = ["BNCI2014001", "BNCI2014002", "BNCI2014004"]

MAIN_MODELS = ["EEGNet", "shallow", "DBConformer", "FastGatedConformer"]
MODEL_LABELS = {
    "EEGNet": "EEGNet",
    "shallow": "ShallowConvNet",
    "DBConformer": "DBConformer",
    "FastGatedConformer": "FastGatedConformer",
}

PARAM_FALLBACKS = {
    ("BNCI2014001", "EEGNet"): 1406,
    ("BNCI2014001", "shallow"): 46084,
    ("BNCI2014001", "DBConformer"): 93747,
    ("BNCI2014001", "FastGatedConformer"): 25107,
    ("BNCI2014002", "EEGNet"): 2658,
    ("BNCI2014002", "shallow"): 31044,
    ("BNCI2014002", "DBConformer"): 251427,
    ("BNCI2014002", "FastGatedConformer"): 45507,
    ("BNCI2014004", "EEGNet"): 1318,
    ("BNCI2014004", "shallow"): 16964,
    ("BNCI2014004", "DBConformer"): 92267,
    ("BNCI2014004", "FastGatedConformer"): 23627,
}

RUNTIME_MINUTES = {
    ("BNCI2014001", "DBConformer"): 54.80,
    ("BNCI2014001", "FastGatedConformer"): 21.65,
    ("BNCI2014002", "DBConformer"): 127.23,
    ("BNCI2014002", "FastGatedConformer"): 40.70,
    ("BNCI2014004", "DBConformer"): 27.44,
    ("BNCI2014004", "FastGatedConformer"): 13.49,
}


def formal_rows(dataset):
    rows = []
    for path in LOG_DIR.glob(f"*{dataset}_*.txt"):
        parsed = parse_log(path)
        if not parsed:
            continue
        if parsed["env"] == "gpu" and parsed["epoch"] == "100":
            rows.append(parsed)
    return rows


def latest_per_seed(rows):
    selected = {}
    for row in sorted(rows, key=lambda r: r["mtime"]):
        key = (
            row["backbone"],
            row["adv"],
            row["branch"],
            row["chn_atten"],
            row["seed"],
        )
        selected[key] = row
    return list(selected.values())


def group_rows(rows):
    grouped = defaultdict(list)
    for row in latest_per_seed(rows):
        key = (row["backbone"], row["adv"], row["branch"], row["chn_atten"])
        grouped[key].append(row)
    return grouped


def summarize(group):
    group = sorted(group, key=lambda r: r["seed"])
    arr = np.array([row["arr"] for row in group], dtype=float)
    seed_means = arr.mean(axis=1)
    params = next((row["params"] for row in reversed(group) if row["params"]), None)
    return {
        "seeds": [row["seed"] for row in group],
        "n": len(group),
        "mean": float(arr.mean()),
        "seed_std": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
        "subject_mean": arr.mean(axis=0),
        "params": params,
    }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_main_accuracy(all_groups):
    rows = []
    subject_rows = []
    for dataset in DATASETS:
        groups = all_groups[dataset]
        for model in MAIN_MODELS:
            key_candidates = [
                key for key in groups
                if key[0] == model and key[1] in {"0", "0.0"} and key[2] == "all" and key[3] == "True"
            ]
            if not key_candidates:
                continue
            summary = summarize(groups[key_candidates[-1]])
            if summary["n"] < 5:
                continue
            rows.append({
                "dataset": dataset,
                "model": MODEL_LABELS[model],
                "mean_accuracy": round(summary["mean"], 3),
                "seed_std": round(summary["seed_std"], 3),
                "n_seeds": summary["n"],
                "seeds": " ".join(map(str, summary["seeds"])),
            })
            for idx, value in enumerate(summary["subject_mean"]):
                subject_rows.append({
                    "dataset": dataset,
                    "model": MODEL_LABELS[model],
                    "subject": f"S{idx}",
                    "mean_accuracy": round(float(value), 3),
                })

    macro = defaultdict(list)
    macro_std = defaultdict(list)
    for row in rows:
        macro[row["model"]].append(float(row["mean_accuracy"]))
        macro_std[row["model"]].append(float(row["seed_std"]))
    for model in [MODEL_LABELS[m] for m in MAIN_MODELS]:
        if len(macro[model]) == len(DATASETS):
            rows.append({
                "dataset": "Macro average",
                "model": model,
                "mean_accuracy": round(float(np.mean(macro[model])), 3),
                "seed_std": round(float(np.mean(macro_std[model])), 3),
                "n_seeds": 5,
                "seeds": "1 2 3 4 5",
            })

    write_csv(
        OUT_DIR / "main_accuracy.csv",
        rows,
        ["dataset", "model", "mean_accuracy", "seed_std", "n_seeds", "seeds"],
    )
    write_csv(
        OUT_DIR / "subject_level_accuracy.csv",
        subject_rows,
        ["dataset", "model", "subject", "mean_accuracy"],
    )


def export_ablation(all_groups):
    ablations = [
        ("Full", "all", "True"),
        ("No channel attention", "all", "False"),
        ("Temporal-only", "temporal", "True"),
    ]
    rows = []
    for dataset in DATASETS:
        groups = all_groups[dataset]
        for label, branch, chn_atten in ablations:
            key_candidates = [
                key for key in groups
                if key[0] == "FastGatedConformer"
                and key[1] in {"0", "0.0"}
                and key[2] == branch
                and key[3] == chn_atten
            ]
            if not key_candidates:
                continue
            summary = summarize(groups[key_candidates[-1]])
            if summary["n"] < 5:
                continue
            rows.append({
                "dataset": dataset,
                "variant": label,
                "mean_accuracy": round(summary["mean"], 3),
                "seed_std": round(summary["seed_std"], 3),
                "n_seeds": summary["n"],
            })
    macro = defaultdict(list)
    macro_std = defaultdict(list)
    for row in rows:
        macro[row["variant"]].append(float(row["mean_accuracy"]))
        macro_std[row["variant"]].append(float(row["seed_std"]))
    for label, _, _ in ablations:
        if len(macro[label]) == len(DATASETS):
            rows.append({
                "dataset": "Macro average",
                "variant": label,
                "mean_accuracy": round(float(np.mean(macro[label])), 3),
                "seed_std": round(float(np.mean(macro_std[label])), 3),
                "n_seeds": 5,
            })

    write_csv(
        OUT_DIR / "ablation_accuracy.csv",
        rows,
        ["dataset", "variant", "mean_accuracy", "seed_std", "n_seeds"],
    )


def export_params():
    rows = []
    for dataset in DATASETS:
        for model in MAIN_MODELS:
            rows.append({
                "dataset": dataset,
                "model": MODEL_LABELS[model],
                "parameters": PARAM_FALLBACKS[(dataset, model)],
            })
    write_csv(OUT_DIR / "parameter_count.csv", rows, ["dataset", "model", "parameters"])


def export_runtime():
    rows = []
    for dataset in DATASETS:
        db = RUNTIME_MINUTES[(dataset, "DBConformer")]
        fast = RUNTIME_MINUTES[(dataset, "FastGatedConformer")]
        rows.append({
            "dataset": dataset,
            "model": "DBConformer",
            "runtime_minutes": db,
            "reduction_vs_dbconformer_percent": 0.0,
        })
        rows.append({
            "dataset": dataset,
            "model": "FastGatedConformer",
            "runtime_minutes": fast,
            "reduction_vs_dbconformer_percent": round((1 - fast / db) * 100, 2),
        })
    write_csv(
        OUT_DIR / "runtime_minutes.csv",
        rows,
        ["dataset", "model", "runtime_minutes", "reduction_vs_dbconformer_percent"],
    )


def export_notes():
    note = """# BSPC Figure Data Notes

These CSV files are generated from the existing training logs and recorded runtime/parameter summaries. They do not require retraining.

## Files

- `main_accuracy.csv`: for the main accuracy comparison figure.
- `subject_level_accuracy.csv`: for subject-level statistical testing or supplementary plots.
- `parameter_count.csv`: for the parameter count comparison figure.
- `runtime_minutes.csv`: for the runtime comparison figure.
- `ablation_accuracy.csv`: for the ablation study figure.

## Manual Figures Still Needed

- Graphical abstract.
- Figure 1 model architecture diagram.

These two figures are conceptual/method figures and should be drawn manually by the author.

## Submission Caution

Use these CSV files as the numerical source. For BSPC submission, create the final figures yourself with Origin, Excel, PowerPoint, MATLAB, Python/matplotlib, or another standard plotting tool. Keep the editable source files.
"""
    (OUT_DIR / "README.md").write_text(note, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_groups = {dataset: group_rows(formal_rows(dataset)) for dataset in DATASETS}
    export_main_accuracy(all_groups)
    export_ablation(all_groups)
    export_params()
    export_runtime()
    export_notes()
    print(f"Exported BSPC figure data to: {OUT_DIR}")


if __name__ == "__main__":
    main()
