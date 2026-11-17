"""Print or run explicit archived training recipes."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "c16_r18": dict(in_dim=512, latent=35, kl=.01, cls=1., patience=10, seeds="6,7,8,9,10"),
    "c16_ctrans": dict(in_dim=768, latent=35, kl=.01, cls=1., patience=10,
                        seeds=",".join(str(s) for s in range(1, 31))),
    "c17_r18": dict(in_dim=512, latent=24, kl=1., cls=100., patience=0,
                     seeds="2021,2022,2023,2024,2025"),
}
VARIANTS = {
    "full": ("adaptive_topk", .05), "topk_only": ("adaptive_topk", 0.),
    "anchor_only": ("max", .05), "max_only": ("max", 0.), "mean": ("mean", 0.),
}


def build_command(args):
    config = RECIPES[args.recipe]
    c17 = args.recipe == "c17_r18"
    pooling, anchor = VARIANTS[args.variant]
    command = [sys.executable, "-m", "maxn_code.maxn_camelyon17" if c17 else "maxn_code.maxn_camelyon16"]
    options = {
        "file_path" if c17 else "dataset_dir": args.data,
        "in_dim": config["in_dim"], "instance_latent_dim": config["latent"],
        "kl_divergence_coef": config["kl"], "aux_loss_multiplier_y": config["cls"],
        "lr" if c17 else "initial_learning_rate": 1e-4, "weight_decay": 1e-4,
        "seeds": args.seeds or config["seeds"], "epochs": 100, "batch_size": 3,
        "early_stop_patience": config["patience"], "train_pooling": pooling, "eval_pooling": "max",
        "topk": 5, "topk_max": 16, "adaptive_gamma": .5, "topk_gamma": 1.,
        "topk_dropout": 0., "feature_noise_sigma": 0., "neg_topk": 5, "neg_coef": 0.,
        "anchor_coef": anchor, "anchor_sim_threshold": args.similarity,
        "anchor_expand_topk": 64, "anchor_min_score": .9,
        "checkpoint_dir": str(Path(args.output).expanduser().resolve() / "checkpoints"),
    }
    if c17:
        if not args.csv:
            raise ValueError("--csv is required for c17_r18")
        options["csv_path"] = args.csv
    for name, value in options.items():
        command.extend(["--" + name, str(value)])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=RECIPES)
    parser.add_argument("--data", required=True)
    parser.add_argument("--csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--similarity", type=float, choices=[.5, .75, .9], default=.75)
    parser.add_argument("--seeds", help="Override the archived repeat set explicitly")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        command = build_command(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        output = Path(args.output).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        with (output / "train.log").open("x") as handle:
            subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
