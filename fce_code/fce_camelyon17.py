import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils

from camelyon17_utils import (
    build_HDF5_feat_dataset,
    camelyon17_collate_fn,
    evaluate_model,
    evaluate_model_with_perclass,
    count_trainable_parameters,
    set_seed,
    parse_seed_list,
    mean_ci,
)
from fce_camelyon16 import FocusmilSingleBranch


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_neg = 0.0
    total_anchor = 0.0
    total_k = 0.0
    n_batches = 0
    for batch in train_loader:
        feats = batch['feats'].to(device)
        bag_idx = batch['bag_idx'].to(device)
        bag_labels = batch['bag_labels'].to(device)

        output = model(feats, bag_idx, bag_labels)
        loss = output['total_loss']

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_neg += float(output.get('neg_loss', torch.tensor(0.0)).item())
        total_anchor += float(output.get('anchor_loss', torch.tensor(0.0)).item())
        total_k += float(output.get('mean_selected_k', 0.0))
        n_batches += 1

    denom = max(1, n_batches)
    return {
        'loss': total_loss / denom,
        'neg': total_neg / denom,
        'anchor': total_anchor / denom,
        'avg_k': total_k / denom,
    }


def training_procedure(args, train_loader, val_loader, device, model_save_path):
    model = FocusmilSingleBranch(
        instance_latent_dim=args.instance_latent_dim,
        in_dim=args.in_dim,
        train_pooling=args.train_pooling,
        eval_pooling=args.eval_pooling,
        topk=args.topk,
        topk_max=args.topk_max,
        adaptive_gamma=args.adaptive_gamma,
        topk_gamma=args.topk_gamma,
        topk_dropout=args.topk_dropout,
        feature_noise_sigma=args.feature_noise_sigma,
        neg_topk=args.neg_topk,
        neg_coef=args.neg_coef,
        anchor_coef=args.anchor_coef,
        anchor_sim_threshold=args.anchor_sim_threshold,
        anchor_expand_topk=args.anchor_expand_topk,
        anchor_min_score=args.anchor_min_score,
    ).to(device)
    model.kl_divergence_coef = args.kl_divergence_coef
    model.aux_loss_multiplier = args.aux_loss_multiplier_y

    n_params = count_trainable_parameters(model)
    print(
        f"[INIT C17 Anchor-Expand] #Params={n_params}, batch_size={args.batch_size}, epochs={args.epochs}, "
        f"train_pooling={args.train_pooling}, eval_pooling={args.eval_pooling}, topk_max={args.topk_max}, "
        f"adaptive_gamma={args.adaptive_gamma}, anchor_coef={args.anchor_coef}, "
        f"anchor_sim_threshold={args.anchor_sim_threshold}, anchor_expand_topk={args.anchor_expand_topk}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc = 0.0
    best_metrics = None
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)

        if val_metrics['bag_auc'] > best_auc:
            best_auc = val_metrics['bag_auc']
            best_metrics = val_metrics
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"[SAVE] best model at epoch={epoch + 1}, auc={best_auc:.4f}")
        else:
            if (epoch + 1) >= args.early_stop_start_epoch:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs}, train_loss={train_stats['loss']:.4f}, "
                f"anchor={train_stats['anchor']:.4f}, avg_k={train_stats['avg_k']:.2f}, "
                f"val_bag_auc={val_metrics['bag_auc']:.4f}"
            )

        early_stop_enabled = (
            args.early_stop_patience > 0
            and (epoch + 1) >= args.early_stop_start_epoch
        )
        if early_stop_enabled and epochs_no_improve >= args.early_stop_patience:
            print(
                f"[EARLY STOP] no val AUC improvement for {args.early_stop_patience} epochs "
                f"after epoch {args.early_stop_start_epoch}; stop at epoch={epoch + 1}"
            )
            break

    if best_metrics is not None:
        model.load_state_dict(torch.load(model_save_path))
    return model, best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--seeds', type=str, default='2021,2022,2023,2024,2025')

    parser.add_argument('--instance_latent_dim', type=int, default=24)
    parser.add_argument('--in_dim', type=int, default=512)
    parser.add_argument('--kl_divergence_coef', type=float, default=1.0)
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=100.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early_stop_patience', type=int, default=0)
    parser.add_argument('--early_stop_start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/c17_fce')

    parser.add_argument('--train_pooling', type=str, default='adaptive_topk', choices=['max', 'fixed_topk', 'adaptive_topk', 'anchor_mean'])
    parser.add_argument('--eval_pooling', type=str, default='max', choices=['max', 'fixed_topk', 'adaptive_topk', 'anchor_mean'])
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--topk_max', type=int, default=16)
    parser.add_argument('--adaptive_gamma', type=float, default=0.5)
    parser.add_argument('--topk_gamma', type=float, default=1.0)
    parser.add_argument('--topk_dropout', type=float, default=0.0)
    parser.add_argument('--feature_noise_sigma', type=float, default=0.0)
    parser.add_argument('--neg_topk', type=int, default=5)
    parser.add_argument('--neg_coef', type=float, default=0.0)
    parser.add_argument('--anchor_coef', type=float, default=0.05)
    parser.add_argument('--anchor_sim_threshold', type=float, default=0.75)
    parser.add_argument('--anchor_expand_topk', type=int, default=64)
    parser.add_argument('--anchor_min_score', type=float, default=0.9)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seeds = parse_seed_list(args.seeds)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print(f'[*] file_path={args.file_path}')
    print(f'[*] csv_path={args.csv_path}')
    print(f'[*] seeds={seeds}')

    all_results = []
    for run_idx, seed in enumerate(seeds, 1):
        print(f"\n>>> RUN {run_idx}/{len(seeds)} | Seed: {seed} <<<")
        set_seed(seed)
        args.seed = seed

        train_dataset, val_dataset, test_dataset = build_HDF5_feat_dataset(args.file_path, args)
        train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.num_workers, collate_fn=camelyon17_collate_fn)
        val_loader = data_utils.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                           num_workers=args.num_workers, collate_fn=camelyon17_collate_fn)
        test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                            num_workers=args.num_workers, collate_fn=camelyon17_collate_fn)

        ckpt = os.path.join(
            args.checkpoint_dir,
            f"best_c17_fce_train-{args.train_pooling}_eval-{args.eval_pooling}_acoef{args.anchor_coef}"
            f"_asim{args.anchor_sim_threshold}_atop{args.anchor_expand_topk}_seed{seed}.pth"
        )
        model, best_val_metrics = training_procedure(args, train_loader, val_loader, device, ckpt)
        print('\n[Validation best metrics]', best_val_metrics)
        test_metrics = evaluate_model_with_perclass(model, test_loader, device)
        print('[Test metrics]', test_metrics)

        result = {
            'seed': seed,
            'validation_auc': best_val_metrics['bag_auc'] if best_val_metrics else None,
            'test_overall_auc': test_metrics['overall']['auc'],
            'test_overall_acc': test_metrics['overall']['acc'],
            'test_overall_f1': test_metrics['overall']['f1'],
            'test_per_cat': test_metrics['per_cat'],
        }
        all_results.append(result)
        print(f"Run {run_idx} Test: auc={result['test_overall_auc']:.4f}, "
              f"acc={result['test_overall_acc']:.4f}, f1={result['test_overall_f1']:.4f}")
        del model
        torch.cuda.empty_cache()

    aucs = [r['test_overall_auc'] for r in all_results]
    accs = [r['test_overall_acc'] for r in all_results]
    f1s = [r['test_overall_f1'] for r in all_results]
    auc_m, auc_l, auc_h = mean_ci(aucs)
    acc_m, acc_l, acc_h = mean_ci(accs)
    f1_m, f1_l, f1_h = mean_ci(f1s)

    print('\n====================================================')
    print('  FINAL CAMELYON17 ANCHOR-EXPAND FOCUSMIL RESULTS')
    print('====================================================')
    print(f'Slide-level AUC : {auc_m:.4f} ({auc_l:.4f}, {auc_h:.4f})')
    print(f'Slide-level ACC : {acc_m:.4f} ({acc_l:.4f}, {acc_h:.4f})')
    print(f'Slide-level F1  : {f1_m:.4f} ({f1_l:.4f}, {f1_h:.4f})')
    print('====================================================')
    print('Per-seed results:')
    for r in all_results:
        print(f"  seed={r['seed']}: auc={r['test_overall_auc']:.4f}, "
              f"acc={r['test_overall_acc']:.4f}, f1={r['test_overall_f1']:.4f}")
    print('[*] All runs completed.')


if __name__ == '__main__':
    main()
