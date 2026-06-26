import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.DBConformer import DBConformer
from models.DGMambaConformer import DGMambaConformer
from models.FastGatedConformer import FastGatedConformer


def make_args():
    return argparse.Namespace(
        data_name="BNCI2014001",
        chn=22,
        patch_size=125,
        time_sample_num=1001,
        class_num=2,
        spa_dim=16,
        gate_flag=False,
        posemb_flag=True,
        branch="all",
        chn_atten_flag=True,
    )


def count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    args = make_args()
    models = {
        "DBConformer": DBConformer(args, emb_size=40, tem_depth=2, chn_depth=2, chn=22, n_classes=2),
        "DGMambaConformer": DGMambaConformer(args, emb_size=40, tem_depth=2, chn_depth=2, chn=22, n_classes=2),
        "FastGatedConformer": FastGatedConformer(args, emb_size=40, tem_depth=2, chn_depth=2, chn=22, n_classes=2),
    }
    for name, model in models.items():
        total, trainable = count(model)
        print(f"{name}: total={total}, trainable={trainable}")


if __name__ == "__main__":
    main()
