import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = ["BNCI2014001", "BNCI2014002", "BNCI2014004"]
DEFAULT_SEEDS = ["1", "2", "3", "4", "5"]


EXPERIMENTS = {
    "fastgate_full": {
        "backbone": "FastGatedConformer",
        "extra": [],
    },
    "fastgate_no_channel_attention": {
        "backbone": "FastGatedConformer",
        "extra": ["--no-channel-attention"],
    },
    "fastgate_temporal_only": {
        "backbone": "FastGatedConformer",
        "extra": ["--branch", "temporal"],
    },
    "dbconformer": {
        "backbone": "DBConformer",
        "extra": [],
    },
}


def build_command(args, experiment_name):
    spec = EXPERIMENTS[experiment_name]
    cmd = [
        sys.executable,
        "DBConformer_LOSO.py",
    ]
    if args.device_id:
        cmd.append(args.device_id)
    cmd.extend([
        "--backbone", spec["backbone"],
        "--data-names", *args.datasets,
        "--seeds", *args.seeds,
        "--max-epoch", str(args.max_epoch),
        "--batch-size", str(args.batch_size),
    ])
    cmd.extend(spec["extra"])
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run a sequence of DBConformer_LOSO experiments with unchanged logging/checkpoint behavior."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["fastgate_no_channel_attention", "fastgate_temporal_only"],
        choices=list(EXPERIMENTS.keys()),
        help="Experiments to run in sequence.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets passed to DBConformer_LOSO.py.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds passed to DBConformer_LOSO.py.",
    )
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device-id",
        default="",
        help="Optional CUDA device id, e.g. 0. Leave empty to let DBConformer_LOSO.py auto-detect CUDA.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    print("Working directory:", project_dir)
    print("Experiments:", ", ".join(args.experiments))
    print("Datasets:", ", ".join(args.datasets))
    print("Seeds:", ", ".join(args.seeds))
    print()

    for idx, experiment_name in enumerate(args.experiments, start=1):
        cmd = build_command(args, experiment_name)
        print("=" * 80)
        print(f"[{idx}/{len(args.experiments)}] {experiment_name}")
        print("Command:", " ".join(cmd))
        print("=" * 80)
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=project_dir)
        if result.returncode != 0:
            raise SystemExit(
                f"Experiment '{experiment_name}' failed with exit code {result.returncode}."
            )

    print()
    print("All requested experiments finished.")


if __name__ == "__main__":
    main()
