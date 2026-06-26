import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "docs" / "bspc_submission" / "figure_data" / "subject_level_accuracy.csv"
OUT_DIR = ROOT / "docs" / "bspc_submission" / "statistics"
OUT_CSV = OUT_DIR / "statistical_tests.csv"
OUT_MD = OUT_DIR / "statistical_tests.md"

PROPOSED = "FastGatedConformer"
BASELINES = ["DBConformer", "EEGNet", "ShallowConvNet"]


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_sided_normal_p(z):
    return max(0.0, min(1.0, 2.0 * (1.0 - normal_cdf(abs(z)))))


def rank_abs(values):
    pairs = sorted((abs(v), idx) for idx, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for _, idx in pairs[i:j]:
            ranks[idx] = avg_rank
        i = j
    return ranks


def wilcoxon_signed_rank(x, y):
    diffs = [a - b for a, b in zip(x, y) if abs(a - b) > 1e-12]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0, 0, 0.0

    ranks = rank_abs(diffs)
    w_pos = sum(r for r, d in zip(ranks, diffs) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, diffs) if d < 0)
    w = min(w_pos, w_neg)

    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w == 0:
        return w, 1.0, n, 0.0
    z = (w - mean_w) / math.sqrt(var_w)
    p = two_sided_normal_p(z)
    return w, p, n, z


def sign_test(x, y):
    pos = sum(1 for a, b in zip(x, y) if a > b)
    neg = sum(1 for a, b in zip(x, y) if a < b)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return pos, neg, min(1.0, 2.0 * prob)


def paired_summary(x, y):
    diffs = [a - b for a, b in zip(x, y)]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    if n > 1:
        sd = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / (n - 1))
    else:
        sd = 0.0
    dz = mean_diff / sd if sd else 0.0
    return mean_diff, sd, dz


def load_rows():
    table = defaultdict(dict)
    with INPUT.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], row["subject"])
            table[key][row["model"]] = float(row["mean_accuracy"])
    return table


def collect_pairs(table, dataset, baseline):
    subjects = []
    proposed_scores = []
    baseline_scores = []
    for (ds, subject), scores in sorted(table.items()):
        if dataset != "All datasets" and ds != dataset:
            continue
        if PROPOSED in scores and baseline in scores:
            subjects.append(f"{ds}:{subject}" if dataset == "All datasets" else subject)
            proposed_scores.append(scores[PROPOSED])
            baseline_scores.append(scores[baseline])
    return subjects, proposed_scores, baseline_scores


def main():
    table = load_rows()
    datasets = sorted({ds for ds, _ in table}) + ["All datasets"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in datasets:
        for baseline in BASELINES:
            subjects, fast, base = collect_pairs(table, dataset, baseline)
            if len(subjects) < 2:
                continue
            mean_diff, sd_diff, dz = paired_summary(fast, base)
            w, wilcoxon_p, n_nonzero, z = wilcoxon_signed_rank(fast, base)
            pos, neg, sign_p = sign_test(fast, base)
            rows.append({
                "dataset": dataset,
                "comparison": f"{PROPOSED} - {baseline}",
                "n_subjects": len(subjects),
                "mean_diff_accuracy": round(mean_diff, 4),
                "sd_diff": round(sd_diff, 4),
                "cohens_dz": round(dz, 4),
                "wilcoxon_w": round(w, 4),
                "wilcoxon_z_approx": round(z, 4),
                "wilcoxon_p_approx": round(wilcoxon_p, 6),
                "sign_positive": pos,
                "sign_negative": neg,
                "sign_p_exact": round(sign_p, 6),
            })

    fields = [
        "dataset",
        "comparison",
        "n_subjects",
        "mean_diff_accuracy",
        "sd_diff",
        "cohens_dz",
        "wilcoxon_w",
        "wilcoxon_z_approx",
        "wilcoxon_p_approx",
        "sign_positive",
        "sign_negative",
        "sign_p_exact",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Statistical Tests",
        "",
        "Input: `docs/bspc_submission/figure_data/subject_level_accuracy.csv`",
        "",
        "The Wilcoxon p-value uses a normal approximation implemented without SciPy. The sign-test p-value is exact under a two-sided binomial test.",
        "",
        "| Dataset | Comparison | n | Mean diff | Wilcoxon p | Sign + / - | Sign p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {row['n_subjects']} | "
            f"{row['mean_diff_accuracy']:.3f} | {row['wilcoxon_p_approx']:.4f} | "
            f"{row['sign_positive']}/{row['sign_negative']} | {row['sign_p_exact']:.4f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
