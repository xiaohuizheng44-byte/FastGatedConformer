import argparse
import ast
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_log(path):
    text = path.read_text(encoding="cp936", errors="ignore")
    seed = re.findall(r"\nSEED:(\d+)", text)
    backbone = re.findall(r"\nbackbone:([^\n]+)", text)
    adv = re.findall(r"\nsubject_adv_weight:([^\n]+)", text)
    env = re.findall(r"\ndata_env:([^\n]+)", text)
    branch = re.findall(r"\nbranch:([^\n]+)", text)
    chn_atten = re.findall(r"\nchn_atten_flag:([^\n]+)", text)
    epoch = re.findall(r"\nmax_epoch:([^\n]+)", text)
    params = re.findall(r"Model params: total=(\d+), trainable=(\d+)", text)
    arrays = []
    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("[") and line.endswith("]") and "," in line):
            continue
        try:
            value = ast.literal_eval(line)
        except Exception:
            continue
        if isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
            arrays.append(value)
    if not arrays or not seed or not backbone:
        return None
    return {
        "file": path.name,
        "seed": int(seed[-1]),
        "backbone": backbone[-1].strip(),
        "adv": adv[-1].strip() if adv else "NA",
        "env": env[-1].strip() if env else "NA",
        "branch": branch[-1].strip() if branch else "NA",
        "chn_atten": chn_atten[-1].strip() if chn_atten else "NA",
        "epoch": epoch[-1].strip() if epoch else "NA",
        "params": int(params[-1][0]) if params else None,
        "arr": arrays[-1],
        "avg": float(np.mean(arrays[-1])),
        "mtime": path.stat().st_mtime,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--dataset", default="BNCI2014001")
    parser.add_argument("--min-runs", type=int, default=5)
    parser.add_argument("--formal-only", action="store_true", default=True)
    args = parser.parse_args()

    rows = []
    for path in Path(args.log_dir).glob(f"*{args.dataset}_*.txt"):
        parsed = parse_log(path)
        if parsed:
            rows.append(parsed)

    groups = defaultdict(list)
    for row in rows:
        if args.formal_only and (row["env"] != "gpu" or row["epoch"] != "100"):
            continue
        key = (row["backbone"], row["adv"], row["env"], row["branch"], row["chn_atten"])
        groups[key].append(row)

    for key, group in sorted(groups.items()):
        latest = {}
        for row in sorted(group, key=lambda r: r["mtime"]):
            latest[row["seed"]] = row
        group = [latest[s] for s in sorted(latest)]
        if len(group) < args.min_runs:
            continue
        arr = np.array([row["arr"] for row in group], dtype=float)
        backbone, adv, env, branch, chn_atten = key
        print(f"\n{backbone} adv={adv} env={env} branch={branch} chn_atten={chn_atten} n={len(group)}")
        print("seeds:", [row["seed"] for row in group])
        print("params:", group[-1]["params"])
        print("seed_means:", np.round(arr.mean(axis=1), 3).tolist())
        print("overall_mean:", round(float(arr.mean()), 3))
        if len(group) > 1:
            print("seed_std:", round(float(arr.mean(axis=1).std(ddof=1)), 3))
            print("subject_means:", np.round(arr.mean(axis=0), 3).tolist())
            print("subject_stds:", np.round(arr.std(axis=0, ddof=1), 3).tolist())


if __name__ == "__main__":
    main()
