import os
import h5py
import datetime  # used for obtaining the current timestamp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.utils.data as data_utils
import scipy.stats as st
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from sklearn.metrics import roc_curve, accuracy_score, f1_score

###########################
# 1) Model Definition (Single-branch Version)
###########################
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim=128, in_dim=512):
        super(VariationalEncoder, self).__init__()
        self.fc_initial = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 200),
        )
        self.mean = nn.Linear(200, latent_dim)
        self.logvar = nn.Linear(200, latent_dim)

    def forward(self, x):
        hidden = self.fc_initial(x)
        mu = self.mean(hidden)
        logvar = self.logvar(hidden)
        return mu, logvar

class AuxiliaryYAdaptiveTopK(nn.Module):
    def __init__(
        self,
        instance_latent_dim=35,
        train_pooling="max",
        eval_pooling=None,
        topk=5,
        topk_max=16,
        adaptive_gamma=0.5,
        topk_gamma=1.0,
        topk_dropout=0.0,
    ):
        super(AuxiliaryYAdaptiveTopK, self).__init__()
        self.fc_ins = nn.Linear(instance_latent_dim, 1)
        self.train_pooling = train_pooling
        self.eval_pooling = eval_pooling if eval_pooling is not None else train_pooling
        self.topk = topk
        self.topk_max = topk_max
        self.adaptive_gamma = adaptive_gamma
        self.topk_gamma = topk_gamma
        self.topk_dropout = topk_dropout

    def _pool_scores(self, loc_ins):
        scores = loc_ins.squeeze(-1)
        if scores.numel() == 0:
            return scores.new_tensor(0.0), 0

        max_score = scores.max()
        pooling = self.train_pooling if self.training else self.eval_pooling
        if pooling == "max":
            return max_score, 1

        if pooling == "fixed_topk":
            k_eff = min(max(1, self.topk), scores.numel())
            top_vals = torch.topk(scores, k=k_eff).values
            kept_vals = top_vals

            if self.training and self.topk_dropout > 0:
                keep_mask = torch.rand_like(top_vals) >= self.topk_dropout
                if not keep_mask.any():
                    keep_mask[0] = True
                kept_vals = top_vals[keep_mask]

            pooled = (1.0 - self.topk_gamma) * max_score + self.topk_gamma * kept_vals.mean()
            return pooled, int(kept_vals.numel())

        if pooling == "adaptive_topk":
            k_scan = min(max(1, self.topk_max), scores.numel())
            top_vals = torch.topk(scores, k=k_scan).values
            spread = top_vals.std(unbiased=False)
            threshold = top_vals[0] - self.adaptive_gamma * spread
            keep_mask = top_vals >= threshold
            if not keep_mask.any():
                keep_mask[0] = True
            kept_vals = top_vals[keep_mask]
            pooled = (1.0 - self.topk_gamma) * max_score + self.topk_gamma * kept_vals.mean()
            return pooled, int(kept_vals.numel())

        if pooling == "anchor_mean":
            # Placeholder. FocusmilSingleBranch overrides this with raw-feature
            # anchor-similarity pooling after it has access to raw patch features.
            return max_score, 1

        raise ValueError(f"Unknown pooling mode: {pooling}")

    def forward(self, z_ins, bag_idx):
        """
        Single-branch scenario:
         - loc_ins: [N, 1] each instance probability
         - M: [B, 1] each bag score
        """
        loc_ins_logits = self.fc_ins(z_ins)  # [N,1]
        loc_ins = torch.sigmoid(loc_ins_logits)
        bags = bag_idx.unique()
        device = z_ins.device
        B = bags.shape[0]

        M = torch.zeros((B, 1), device=device)
        selected_ks = []
        for i, bag_id in enumerate(bags):
            idxs_bag = (bag_idx == bag_id).nonzero(as_tuple=True)[0]
            if idxs_bag.numel() > 0:
                pooled, selected_k = self._pool_scores(loc_ins[idxs_bag])
                M[i, :] = pooled
                selected_ks.append(selected_k)
            else:
                M[i, :] = 0.0

        mean_selected_k = float(sum(selected_ks) / len(selected_ks)) if selected_ks else 0.0
        return M, loc_ins, loc_ins_logits, mean_selected_k

class FocusmilSingleBranch(nn.Module):
    """
    Simplified: only single-branch, without diff_loss or third_loss.
    """
    def __init__(
        self,
        instance_latent_dim=128,
        in_dim=512,
        train_pooling="max",
        eval_pooling=None,
        topk=5,
        topk_max=16,
        adaptive_gamma=0.5,
        topk_gamma=1.0,
        topk_dropout=0.0,
        feature_noise_sigma=0.0,
        neg_topk=5,
        neg_coef=0.0,
        anchor_coef=0.05,
        anchor_sim_threshold=0.75,
        anchor_expand_topk=64,
        anchor_min_score=0.9,
    ):
        super(FocusmilSingleBranch, self).__init__()
        self.encoder = VariationalEncoder(latent_dim=instance_latent_dim, in_dim=in_dim)
        self.aux_y = AuxiliaryYAdaptiveTopK(
            instance_latent_dim=instance_latent_dim,
            train_pooling=train_pooling,
            eval_pooling=eval_pooling,
            topk=topk,
            topk_max=topk_max,
            adaptive_gamma=adaptive_gamma,
            topk_gamma=topk_gamma,
            topk_dropout=topk_dropout,
        )
        self.feature_noise_sigma = feature_noise_sigma
        self.neg_topk = neg_topk
        self.neg_coef = neg_coef
        self.anchor_coef = anchor_coef
        self.anchor_sim_threshold = anchor_sim_threshold
        self.anchor_expand_topk = anchor_expand_topk
        self.anchor_min_score = anchor_min_score

        # Hyperparameters set externally
        self.kl_divergence_coef = 1.0
        self.aux_loss_multiplier = 1.0

        # Main classification loss
        # You may use BCEWithLogitsLoss or BCELoss, depending on your need
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, bag, bag_idx, bag_label):
        """
        Training forward:
          - Perform VAE encoding -> rsample() for each instance
          - Obtain instance-level scores (loc_ins) and bag-level score (M)
          - KL + main classification loss
        """
        bag_label_2d = bag_label.view(-1,1).float()  # [B]->[B,1]
        raw_bag = bag

        if self.training and self.feature_noise_sigma > 0:
            bag = bag + self.feature_noise_sigma * torch.randn_like(bag)

        # VAE encoder
        instance_mu, instance_logvar = self.encoder(bag)
        instance_std = (instance_logvar * 0.5).exp_()
        qzx = dist.Normal(instance_mu, instance_std)
        z_ins = qzx.rsample()  # random sampling

        # Single-branch bag score
        M, loc_ins, loc_ins_logits, mean_selected_k = self.aux_y(z_ins, bag_idx)
        if self._use_anchor_mean_pooling():
            M, mean_selected_k = self._anchor_mean_pool_scores(loc_ins, bag_idx, raw_bag)

        # KL
        KL_loss = 0.5 * (
            instance_mu.pow(2) + instance_std.pow(2)
            - 2*torch.log(instance_std + 1e-8) - 1
        ).mean()

        # Main classification loss
        sub_loss = self.bce_loss(M, bag_label_2d)
        neg_loss = self._negative_topk_loss(loc_ins_logits, bag_idx, bag_label)
        anchor_loss = self._anchor_expansion_loss(loc_ins_logits, loc_ins, bag_idx, bag_label, raw_bag)

        total_loss = (
            self.kl_divergence_coef * KL_loss
            + self.aux_loss_multiplier * sub_loss
            + self.aux_loss_multiplier * self.neg_coef * neg_loss
            + self.aux_loss_multiplier * self.anchor_coef * anchor_loss
        )

        return {
            'total_loss': total_loss,
            'KL_loss': KL_loss,
            'sub_loss': sub_loss,
            'neg_loss': neg_loss,
            'anchor_loss': anchor_loss,
            'topk_dropout': self.aux_y.topk_dropout,
            'mean_selected_k': mean_selected_k,
        }

    def _negative_topk_loss(self, loc_ins_logits, bag_idx, bag_label):
        if self.neg_coef <= 0 or self.neg_topk <= 0:
            return loc_ins_logits.new_tensor(0.0)

        hard_logits = []
        logits_1d = loc_ins_logits.squeeze(-1)
        for bag_id in bag_idx.unique():
            bag_id_int = int(bag_id.item())
            if bag_label[bag_id_int].item() > 0.5:
                continue
            idxs_bag = (bag_idx == bag_id).nonzero(as_tuple=True)[0]
            if idxs_bag.numel() == 0:
                continue
            k_eff = min(max(1, self.neg_topk), idxs_bag.numel())
            hard_logits.append(torch.topk(logits_1d[idxs_bag], k=k_eff).values)

        if not hard_logits:
            return loc_ins_logits.new_tensor(0.0)

        hard_logits = torch.cat(hard_logits, dim=0)
        hard_targets = torch.zeros_like(hard_logits)
        return F.binary_cross_entropy_with_logits(hard_logits, hard_targets)

    def _current_pooling_mode(self):
        return self.aux_y.train_pooling if self.training else self.aux_y.eval_pooling

    def _use_anchor_mean_pooling(self):
        return self._current_pooling_mode() == "anchor_mean"

    def _anchor_mean_pool_scores(self, loc_ins, bag_idx, raw_feat):
        scores_1d = loc_ins.squeeze(-1)
        bags = bag_idx.unique()
        M = torch.zeros((bags.shape[0], 1), device=loc_ins.device)
        selected_ks = []

        for i, bag_id in enumerate(bags):
            idxs_bag = (bag_idx == bag_id).nonzero(as_tuple=True)[0]
            if idxs_bag.numel() == 0:
                continue

            bag_scores = scores_1d[idxs_bag]
            anchor_local = int(torch.argmax(bag_scores).item())
            bag_feat = F.normalize(raw_feat[idxs_bag], p=2, dim=1)
            anchor_feat = bag_feat[anchor_local:anchor_local + 1]
            sim_vals = torch.matmul(bag_feat, anchor_feat.t()).squeeze(1)

            topm = min(max(1, self.aux_y.topk_max), bag_scores.numel())
            candidate_local = torch.topk(bag_scores, k=topm).indices
            candidate_sim = sim_vals[candidate_local]
            keep_mask = candidate_sim >= self.anchor_sim_threshold
            if keep_mask.any():
                candidate_local = candidate_local[keep_mask]
                candidate_sim = candidate_sim[keep_mask]
            else:
                candidate_local = torch.tensor([anchor_local], device=loc_ins.device, dtype=torch.long)
                candidate_sim = sim_vals[candidate_local]

            k_eff = min(max(1, self.anchor_expand_topk), candidate_local.numel())
            if candidate_local.numel() > k_eff:
                _, order = torch.topk(candidate_sim, k=k_eff)
                candidate_local = candidate_local[order]

            M[i, 0] = bag_scores[candidate_local].mean()
            selected_ks.append(int(candidate_local.numel()))

        mean_selected_k = float(sum(selected_ks) / len(selected_ks)) if selected_ks else 0.0
        return M, mean_selected_k

    def _anchor_expansion_loss(self, loc_ins_logits, loc_ins, bag_idx, bag_label, raw_feat):
        if self.anchor_coef <= 0 or self.anchor_expand_topk <= 0:
            return loc_ins_logits.new_tensor(0.0)

        selected_logits = []
        logits_1d = loc_ins_logits.squeeze(-1)
        scores_1d = loc_ins.squeeze(-1)

        for bag_id in bag_idx.unique():
            bag_id_int = int(bag_id.item())
            if bag_label[bag_id_int].item() <= 0.5:
                continue

            idxs_bag = (bag_idx == bag_id).nonzero(as_tuple=True)[0]
            if idxs_bag.numel() == 0:
                continue

            bag_scores = scores_1d[idxs_bag]
            anchor_local = int(torch.argmax(bag_scores).item())
            anchor_score = bag_scores[anchor_local].detach()
            if anchor_score.item() < self.anchor_min_score:
                continue

            bag_feat = F.normalize(raw_feat[idxs_bag], p=2, dim=1)
            anchor_feat = bag_feat[anchor_local:anchor_local + 1]
            sim_vals = torch.matmul(bag_feat, anchor_feat.t()).squeeze(1)
            candidate_local = (sim_vals >= self.anchor_sim_threshold).nonzero(as_tuple=True)[0]
            if candidate_local.numel() == 0:
                continue

            k_eff = min(max(1, self.anchor_expand_topk), candidate_local.numel())
            if candidate_local.numel() > k_eff:
                _, order = torch.topk(sim_vals[candidate_local], k=k_eff)
                candidate_local = candidate_local[order]

            selected_logits.append(logits_1d[idxs_bag[candidate_local]])

        if not selected_logits:
            return loc_ins_logits.new_tensor(0.0)

        selected_logits = torch.cat(selected_logits, dim=0)
        positive_targets = torch.ones_like(selected_logits)
        return F.binary_cross_entropy_with_logits(selected_logits, positive_targets)


    @torch.no_grad()
    def forward_no_sampling(self, bag, bag_idx):
        """
        Testing/inference forward:
         - No random sampling (directly use mu)
         - Obtain bag-level score M and instance-level score loc_ins
        """
        instance_mu, instance_logvar = self.encoder(bag)
        instance_std = (instance_logvar * 0.5).exp_()
        # no sampling
        z_ins = instance_mu
        M, loc_ins, _, _ = self.aux_y(z_ins, bag_idx)
        if self._use_anchor_mean_pooling():
            M, _ = self._anchor_mean_pool_scores(loc_ins, bag_idx, bag)

        KL_loss = 0.5 * (
            instance_mu.pow(2) + instance_std.pow(2)
            - 2*torch.log(instance_std + 1e-8) - 1
        ).mean()

        return M, loc_ins, KL_loss

    @torch.no_grad()
    def predict_instance_score(self, bag, bag_idx):
        """
        Return instance-level scores [N]
        """
        M, loc_ins, KL_loss = self.forward_no_sampling(bag, bag_idx)
        return loc_ins.squeeze(-1)

    @torch.no_grad()
    def predict_bag_score(self, bag, bag_idx):
        """
        Return bag-level scores [B]
        """
        M, loc_ins, KL_loss = self.forward_no_sampling(bag, bag_idx)
        return M.squeeze(-1)


###########################
# 2) Dataset Definition & collate_fn (unchanged)
###########################
class CAMELYON_16_10x_feat(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', return_bag=False):
        self.root_dir = root_dir
        self.split = split
        self.return_bag = return_bag

        # Load features and labels
        self.feat_file = os.path.join(self.root_dir, f"{self.split}_patch_feat.h5")
        with h5py.File(self.feat_file, 'r') as h5f:
            self.all_patches = np.array(h5f['dataset_1'])

        self.patch_label = np.load(os.path.join(self.root_dir, f"{self.split}_patch_label.npy"))
        self.patch_corresponding_slide_label = np.load(os.path.join(self.root_dir, f"{self.split}_patch_corresponding_slide_label.npy"))
        self.patch_corresponding_slide_index = np.load(os.path.join(self.root_dir, f"{self.split}_patch_corresponding_slide_index.npy"))
        self.patch_corresponding_slide_name = np.load(os.path.join(self.root_dir, f"{self.split}_patch_corresponding_slide_name.npy"))

        self.num_patches = self.all_patches.shape[0]
        self.num_slides = self.patch_corresponding_slide_index.max() + 1
        print(f"[DATA INFO] {self.split} => num_slides={self.num_slides}, num_patches={self.num_patches}")

        self.slide_feat_all = []
        self.slide_label_all = []
        self.slide_patch_label_all = []
        for i in range(self.num_slides):
            idx_from_same_slide = np.nonzero(self.patch_corresponding_slide_index == i)[0]
            self.slide_feat_all.append(self.all_patches[idx_from_same_slide])
            # Ensure that the label within the same slide is consistent
            if self.patch_corresponding_slide_label[idx_from_same_slide].max() != \
               self.patch_corresponding_slide_label[idx_from_same_slide].min():
                raise ValueError("Inconsistent slide labels in the same slide!")
            self.slide_label_all.append(self.patch_corresponding_slide_label[idx_from_same_slide].max())
            self.slide_patch_label_all.append(self.patch_label[idx_from_same_slide])

    def __len__(self):
        if self.return_bag:
            return self.num_slides
        else:
            return self.num_patches

    def __getitem__(self, index):
        if self.return_bag:
            bag = torch.tensor(self.slide_feat_all[index], dtype=torch.float32)
            slide_label = self.slide_label_all[index]
            patch_labels = self.slide_patch_label_all[index]
            return bag, patch_labels, slide_label, index
        else:
            patch_feat = torch.tensor(self.all_patches[index], dtype=torch.float32)
            patch_label = self.patch_label[index]
            slide_label = self.patch_corresponding_slide_label[index]
            slide_idx = self.patch_corresponding_slide_index[index]
            return patch_feat, patch_label, slide_label, slide_idx

def mi_collate_img(batch):
    all_bags = []
    all_bag_labels = []
    all_instance_labels = []
    bag_indices = []
    for i, (bag, patch_labels, slide_label, idx_slide) in enumerate(batch):
        all_bags.append(bag)
        all_instance_labels.append(torch.tensor(patch_labels, dtype=torch.float32))
        all_bag_labels.append(slide_label)
        bag_indices.append(torch.full((bag.shape[0],), i, dtype=torch.long))

    all_bags_cat = torch.cat(all_bags, dim=0)
    bag_idx_cat = torch.cat(bag_indices, dim=0)
    all_bag_labels = torch.tensor(all_bag_labels, dtype=torch.float32)
    all_ins_labels = torch.cat(all_instance_labels, dim=0)
    return all_bags_cat, bag_idx_cat, all_bag_labels, all_ins_labels


###########################
# 3) Validation/Evaluation Function (unchanged)
###########################
def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    slide_auroc = BinaryAUROC().to(device)
    patch_auroc = BinaryAUROC().to(device)
    patch_auprc = BinaryAveragePrecision().to(device)

    all_patch_probs = []
    all_patch_labels = []
    all_slide_probs = []
    all_slide_labels = []

    for batch in dataloader:
        bag, bag_idx, bag_label, ins_label = batch
        bag, bag_idx = bag.to(device), bag_idx.to(device)
        bag_label = bag_label.to(device).long()
        ins_label = ins_label.to(device).long()

        patch_preds = model.predict_instance_score(bag, bag_idx)
        slide_preds = model.predict_bag_score(bag, bag_idx)

        slide_auroc.update(slide_preds, bag_label)
        patch_auroc.update(patch_preds, ins_label)
        patch_auprc.update(patch_preds, ins_label)

        all_patch_probs.append(patch_preds.cpu().numpy())
        all_patch_labels.append(ins_label.cpu().numpy())
        all_slide_probs.append(slide_preds.cpu().numpy())
        all_slide_labels.append(bag_label.cpu().numpy())

    # 1) AUC
    slide_auc = slide_auroc.compute().item()
    patch_auc = patch_auroc.compute().item()
    patch_auprc_val = patch_auprc.compute().item()

    slide_auroc.reset()
    patch_auroc.reset()
    patch_auprc.reset()

    # 2) F1/ACC
    all_patch_probs = np.concatenate(all_patch_probs, axis=0)
    all_patch_labels = np.concatenate(all_patch_labels, axis=0)
    all_slide_probs = np.concatenate(all_slide_probs, axis=0)
    all_slide_labels = np.concatenate(all_slide_labels, axis=0)

    fpr, tpr, thresholds_patch = roc_curve(all_patch_labels, all_patch_probs)
    _, _, thr_patch = optimal_thresh(fpr, tpr, thresholds_patch)
    patch_preds_bin = (all_patch_probs >= thr_patch).astype(int)
    patch_f1 = f1_score(all_patch_labels, patch_preds_bin)

    fpr2, tpr2, thresholds_slide = roc_curve(all_slide_labels, all_slide_probs)
    _, _, thr_slide = optimal_thresh(fpr2, tpr2, thresholds_slide)
    slide_preds_bin = (all_slide_probs >= thr_slide).astype(int)
    slide_acc = accuracy_score(all_slide_labels, slide_preds_bin)
    slide_f1 = f1_score(all_slide_labels, slide_preds_bin)

    return {
        'slide_auc': slide_auc,
        'slide_f1': slide_f1,
        'slide_acc': slide_acc,

        'patch_auc': patch_auc,
        'patch_auprc': patch_auprc_val,
        'patch_f1': patch_f1,

        'thr_patch': thr_patch,
        'thr_slide': thr_slide
    }


###########################
# 4) Training Procedure (per epoch)
###########################
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_seeds(seed_text):
    return [int(item.strip()) for item in seed_text.split(',') if item.strip()]


def training_procedure(FLAGS, train_loader, val_loader, device, model_save_path):
    """
    After each epoch, test slide_auc on val_loader. If improved, save to model_save_path.
    Return: (model, best_val_metrics)
    """
    # === Initialize single-branch model ===
    model = FocusmilSingleBranch(
        instance_latent_dim=FLAGS.instance_latent_dim,
        in_dim=FLAGS.in_dim,
        train_pooling=FLAGS.train_pooling,
        eval_pooling=FLAGS.eval_pooling,
        topk=FLAGS.topk,
        topk_max=FLAGS.topk_max,
        adaptive_gamma=FLAGS.adaptive_gamma,
        topk_gamma=FLAGS.topk_gamma,
        topk_dropout=FLAGS.topk_dropout,
        feature_noise_sigma=FLAGS.feature_noise_sigma,
        neg_topk=FLAGS.neg_topk,
        neg_coef=FLAGS.neg_coef,
        anchor_coef=FLAGS.anchor_coef,
        anchor_sim_threshold=FLAGS.anchor_sim_threshold,
        anchor_expand_topk=FLAGS.anchor_expand_topk,
        anchor_min_score=FLAGS.anchor_min_score,
    ).to(device)

    # Set coefficients
    model.kl_divergence_coef = FLAGS.kl_divergence_coef
    model.aux_loss_multiplier = FLAGS.aux_loss_multiplier_y

    n_params = count_trainable_parameters(model)
    print(
        f" => [Anchor-Expand TopK FocusMIL] #Params={n_params}, batch_size={FLAGS.batch_size}, "
        f"epochs={FLAGS.epochs}, train_pooling={FLAGS.train_pooling}, "
        f"eval_pooling={FLAGS.eval_pooling}, topk={FLAGS.topk}, topk_max={FLAGS.topk_max}, "
        f"adaptive_gamma={FLAGS.adaptive_gamma}, topk_gamma={FLAGS.topk_gamma}, "
        f"topk_dropout={FLAGS.topk_dropout}, feature_noise_sigma={FLAGS.feature_noise_sigma}, "
        f"neg_topk={FLAGS.neg_topk}, neg_coef={FLAGS.neg_coef}, "
        f"anchor_coef={FLAGS.anchor_coef}, anchor_sim_threshold={FLAGS.anchor_sim_threshold}, "
        f"anchor_expand_topk={FLAGS.anchor_expand_topk}, anchor_min_score={FLAGS.anchor_min_score}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.initial_learning_rate, weight_decay=FLAGS.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    best_slide_auc = 0.0
    best_val_metrics = None
    epochs_without_improve = 0

    for epoch in range(FLAGS.epochs):
        model.train()
        total_loss_this_epoch = 0.
        total_neg_loss = 0.
        total_anchor_loss = 0.
        total_selected_k = 0.
        for batch in train_loader:
            bag, bag_idx, bag_label, ins_label = batch
            bag = bag.to(device)
            bag_idx = bag_idx.to(device)
            bag_label = bag_label.to(device)

            optimizer.zero_grad()
            out = model(bag, bag_idx, bag_label)
            loss_val = out['total_loss']
            loss_val.backward()
            nn.utils.clip_grad_norm_(model.parameters(), FLAGS.grad_clip_norm)
            optimizer.step()
            total_loss_this_epoch += loss_val.item()
            total_neg_loss += out.get('neg_loss', torch.tensor(0.0, device=bag.device)).item()
            total_anchor_loss += out.get('anchor_loss', torch.tensor(0.0, device=bag.device)).item()
            total_selected_k += out.get('mean_selected_k', 0.0)

        avg_loss = total_loss_this_epoch / len(train_loader)
        avg_neg_loss = total_neg_loss / len(train_loader)
        avg_anchor_loss = total_anchor_loss / len(train_loader)
        avg_selected_k = total_selected_k / len(train_loader)
        scheduler.step(avg_loss)

        # === Evaluate on validation set ===
        metrics_val = evaluate_model(model, val_loader, device)
        if metrics_val['slide_auc'] > best_slide_auc:
            best_slide_auc = metrics_val['slide_auc']
            best_val_metrics = metrics_val
            epochs_without_improve = 0
            # Save the current best model
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_without_improve += 1

        if FLAGS.early_stop_patience > 0 and epochs_without_improve >= FLAGS.early_stop_patience:
            print(
                f"   [EarlyStop] epoch={epoch+1}, best_val_slide_auc={best_slide_auc:.4f}, "
                f"patience={FLAGS.early_stop_patience}"
            )
            break

        if (epoch+1) % 10 == 0:
            print(
                f"   [Epoch {epoch+1}/{FLAGS.epochs}] train_loss={avg_loss:.4f}, "
                f"neg={avg_neg_loss:.4f}, anchor={avg_anchor_loss:.4f}, "
                f"avg_kept_topk={avg_selected_k:.2f}, "
                f"val_slide_auc={metrics_val['slide_auc']:.4f}, "
                f"val_patch_auprc={metrics_val['patch_auprc']:.4f}"
            )

    # === After training, load the best model ===
    if best_val_metrics is not None:
        model.load_state_dict(torch.load(model_save_path))
    return model, best_val_metrics


###########################
# 5) Main function: grid-search example
###########################
if __name__ == '__main__':
    import argparse
    import datetime

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--cuda', action='store_true', default=True)

   
    parser.add_argument('--instance_latent_dim', type=int, default=35)
    parser.add_argument('--in_dim', type=int, default=512)
    parser.add_argument('--initial_learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--kl_divergence_coef', type=float, default=0.01)
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=1.0)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early_stop_patience', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_anchor_expand_c16')
    parser.add_argument('--pooling', type=str, default=None, choices=['max', 'fixed_topk', 'adaptive_topk', 'anchor_mean'],
                        help='Deprecated alias for --train_pooling.')
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
    parser.add_argument('--seeds', type=str, default='1,11,111,1111,11111')

    FLAGS = parser.parse_args()
    os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)
    if FLAGS.pooling is not None:
        FLAGS.train_pooling = FLAGS.pooling

    device = torch.device('cuda' if (FLAGS.cuda and torch.cuda.is_available()) else 'cpu')
    print("[*] Device:", device)

   
    full_train_dataset = CAMELYON_16_10x_feat(root_dir=FLAGS.dataset_dir, split='train', return_bag=True)
    test_dataset = CAMELYON_16_10x_feat(root_dir=FLAGS.dataset_dir, split='val', return_bag=True)
    
    test_loader = data_utils.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=mi_collate_img
    )

   
    seeds = parse_seeds(FLAGS.seeds)
    history = {'slide_auc': [], 'slide_acc': [], 'slide_f1': [], 'patch_auc': [], 'patch_auprc': [], 'patch_f1': []}

    for run_idx, seed in enumerate(seeds):
        print(f"\n>>> RUN {run_idx+1}/{len(seeds)} | Seed: {seed} <<<")
        
        
        train_size = int(0.8 * len(full_train_dataset))
        val_size   = len(full_train_dataset) - train_size
        train_split, val_split = torch.utils.data.random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )

        train_loader = data_utils.DataLoader(
            train_split, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4, collate_fn=mi_collate_img
        )
        val_loader = data_utils.DataLoader(
            val_split, batch_size=1, shuffle=False, num_workers=4, collate_fn=mi_collate_img
        )

       
        model_save_path = os.path.join(
            FLAGS.checkpoint_dir,
            f"best_focusmil_anchor_expand_train-{FLAGS.train_pooling}"
            f"_eval-{FLAGS.eval_pooling}_neg{FLAGS.neg_coef}"
            f"_acoef{FLAGS.anchor_coef}_asim{FLAGS.anchor_sim_threshold}"
            f"_atop{FLAGS.anchor_expand_topk}_seed{seed}.pth"
        )

        
        best_model, _ = training_procedure(FLAGS, train_loader, val_loader, device, model_save_path)

       
        test_metrics = evaluate_model(best_model, test_loader, device)

       
        history['slide_auc'].append(test_metrics['slide_auc'])
        history['slide_acc'].append(test_metrics['slide_acc'])
        history['slide_f1'].append(test_metrics['slide_f1'])
        history['patch_auc'].append(test_metrics['patch_auc'])
        history['patch_auprc'].append(test_metrics['patch_auprc'])
        history['patch_f1'].append(test_metrics['patch_f1'])

        print(
            f"Run {run_idx+1} Test: slide_auc={test_metrics['slide_auc']:.4f}, "
            f"slide_acc={test_metrics['slide_acc']:.4f}, "
            f"slide_f1={test_metrics['slide_f1']:.4f}, "
            f"patch_auc={test_metrics['patch_auc']:.4f}, "
            f"patch_auprc={test_metrics['patch_auprc']:.4f}, "
            f"patch_f1={test_metrics['patch_f1']:.4f}"
        )

   
    def print_final_stats(name, data):
        data = np.array(data)
        mean = np.mean(data)
        if len(data) == 1:
            print(f"{name}: {mean:.4f}")
            return
        conf_interval = st.t.interval(0.95, len(data)-1, loc=mean, scale=st.sem(data))
        print(f"{name}: {mean:.4f} ({conf_interval[0]:.4f}, {conf_interval[1]:.4f})")

    print("\n" + "="*52)
    print("  FINAL ANCHOR-EXPAND TOPK FOCUSMIL RESULTS")
    print("="*52)
    print_final_stats("Slide-level AUC ", history['slide_auc'])
    print_final_stats("Slide-level ACC ", history['slide_acc'])
    print_final_stats("Slide-level F1  ", history['slide_f1'])
    print_final_stats("Patch-level AUC  ", history['patch_auc'])
    print_final_stats("Patch-level AUPRC", history['patch_auprc'])
    print_final_stats("Patch-level F1   ", history['patch_f1'])
    print("="*52)
    print("[*] All runs completed.")
