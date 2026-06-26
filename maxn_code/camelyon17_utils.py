import random

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torchmetrics.classification import BinaryAUROC


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_seed_list(seed_text):
    if isinstance(seed_text, int):
        return [seed_text]
    return [int(x.strip()) for x in str(seed_text).split(',') if x.strip()]


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) <= 1:
        return mean, mean, mean
    ci = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean, mean - ci, mean + ci


def split_dataset_camelyon17(file_path, conf):
    """Build the Camelyon17 center-level split used by the paper scripts.

    The metadata CSV must contain `slide_id` and `center`. Slides with
    zero-indexed center >= 3 are used as the held-out test set; the remaining
    centers are split into train/validation with an 8:2 ratio using `conf.seed`.
    Labels are binarized as normal (0) vs tumor (1/2/3).
    """
    df = pd.read_csv(conf.csv_path)
    df.set_index('slide_id', inplace=True)

    h5_data = h5py.File(file_path, 'r')
    slide_names = list(h5_data.keys())

    test_names = []
    train_val_names = []
    for name in slide_names:
        if name not in df.index:
            print(f"[WARN] Slide {name} not in CSV. Skip it.")
            continue
        center_val = int(df.loc[name]['center'])
        if center_val >= 3:
            test_names.append(name)
        else:
            train_val_names.append(name)

    train_names, val_names = train_test_split(
        train_val_names, test_size=0.2, random_state=conf.seed
    )

    def fill_split_dict(names_list):
        split = {}
        for slide_id in names_list:
            slide = h5_data[slide_id]
            original_label = slide.attrs['label']
            label_2class = 0 if original_label == 0 else 1
            split[slide_id] = {
                'input': slide['feat'][:],
                'coords': slide['coords'][:],
                'label': label_2class,
                'label_orig': original_label,
            }
        return split

    train_split = fill_split_dict(train_names)
    val_split = fill_split_dict(val_names)
    test_split = fill_split_dict(test_names)
    h5_data.close()

    print(
        '[INFO] Camelyon17 split =>',
        f'Train={len(train_names)}, Val={len(val_names)}, Test={len(test_names)}',
    )
    return train_split, train_names, val_split, val_names, test_split, test_names


class HDF5FeatDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, data_names):
        super().__init__()
        self.data_dict = data_dict
        self.data_names = data_names

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index):
        slide_id = self.data_names[index]
        data_item = self.data_dict[slide_id]
        return {
            'feat': torch.tensor(data_item['input'], dtype=torch.float32),
            'label': torch.tensor(data_item['label'], dtype=torch.long),
            'coords': data_item['coords'],
            'slide_id': slide_id,
            'label_orig': data_item['label_orig'],
        }


def build_HDF5_feat_dataset(file_path, conf):
    train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_camelyon17(file_path, conf)
    return (
        HDF5FeatDataset(train_split, train_names),
        HDF5FeatDataset(val_split, val_names),
        HDF5FeatDataset(test_split, test_names),
    )


def optimal_thresh(fpr, tpr, thresholds):
    youden = tpr - fpr
    idx = np.argmax(youden)
    return thresholds[idx]


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    slide_auroc = BinaryAUROC().to(device)
    all_bag_scores = []
    all_labels = []

    for batch in dataloader:
        feats = batch['feats'].to(device)
        bag_idx = batch['bag_idx'].to(device)
        bag_labels = batch['bag_labels'].to(device)
        bag_score = model.predict_bag_score(feats, bag_idx)
        slide_auroc.update(bag_score, bag_labels.float())
        all_bag_scores.append(bag_score.cpu().numpy())
        all_labels.append(bag_labels.cpu().numpy())

    slide_auc = slide_auroc.compute().item()
    slide_auroc.reset()
    all_bag_scores = np.concatenate(all_bag_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    fpr, tpr, thresholds = roc_curve(all_labels, all_bag_scores)
    best_thr = optimal_thresh(fpr, tpr, thresholds)
    preds_bin = (all_bag_scores >= best_thr).astype(int)
    return {
        'bag_auc': slide_auc,
        'bag_f1': f1_score(all_labels, preds_bin),
        'bag_acc': accuracy_score(all_labels, preds_bin),
        'best_thr': best_thr,
    }


@torch.no_grad()
def evaluate_model_with_perclass(model, dataloader, device):
    model.eval()
    slide_auroc = BinaryAUROC().to(device)
    all_scores = []
    all_labels_bin = []
    all_label_orig = []

    for batch in dataloader:
        feats = batch['feats'].to(device)
        bag_idx = batch['bag_idx'].to(device)
        bag_labels = batch['bag_labels'].to(device)
        label_orig_list = batch['label_orig_list']
        bag_score = model.predict_bag_score(feats, bag_idx)
        slide_auroc.update(bag_score, bag_labels.float())
        all_scores.append(bag_score.cpu().numpy())
        all_labels_bin.append(bag_labels.cpu().numpy())
        all_label_orig.extend(label_orig_list)

    auc_val = slide_auroc.compute().item()
    slide_auroc.reset()
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels_bin = np.concatenate(all_labels_bin, axis=0)
    all_label_orig = np.array(all_label_orig)

    fpr, tpr, thresholds = roc_curve(all_labels_bin, all_scores)
    best_thr = optimal_thresh(fpr, tpr, thresholds)
    preds_bin = (all_scores >= best_thr).astype(int)
    results_dict = {
        'overall': {
            'auc': auc_val,
            'acc': accuracy_score(all_labels_bin, preds_bin),
            'f1': f1_score(all_labels_bin, preds_bin),
            'best_thr': best_thr,
        },
        'per_cat': {},
    }

    for cat in [1, 2, 3]:
        idx_sub = np.where((all_label_orig == 0) | (all_label_orig == cat))[0]
        if len(idx_sub) == 0:
            results_dict['per_cat'][cat] = f'No samples for cat={cat}'
            continue
        sub_labels = np.where(all_label_orig[idx_sub] == cat, 1, 0)
        if np.sum(sub_labels == 1) == 0:
            results_dict['per_cat'][cat] = f'No positive samples for cat={cat}'
            continue
        sub_scores = all_scores[idx_sub]
        fpr_sub, tpr_sub, thresholds_sub = roc_curve(sub_labels, sub_scores)
        best_thr_sub = optimal_thresh(fpr_sub, tpr_sub, thresholds_sub)
        preds_sub = (sub_scores >= best_thr_sub).astype(int)
        results_dict['per_cat'][cat] = {
            'auc': roc_auc_score(sub_labels, sub_scores),
            'acc': accuracy_score(sub_labels, preds_sub),
            'f1': f1_score(sub_labels, preds_sub),
            'best_thr': best_thr_sub,
        }
    return results_dict


def camelyon17_collate_fn(batch):
    feats_list = []
    bag_idx_list = []
    bag_label_list = []
    coords_list = []
    slide_id_list = []
    label_orig_list = []

    for i, data_dict in enumerate(batch):
        feat_tensor = data_dict['feat']
        feats_list.append(feat_tensor)
        bag_idx_list.append(torch.full((feat_tensor.shape[0],), i, dtype=torch.long))
        bag_label_list.append(data_dict['label'])
        coords_list.append(data_dict['coords'])
        slide_id_list.append(data_dict['slide_id'])
        label_orig_list.append(data_dict['label_orig'])

    return {
        'feats': torch.cat(feats_list, dim=0),
        'bag_idx': torch.cat(bag_idx_list, dim=0),
        'bag_labels': torch.stack(bag_label_list, dim=0).long(),
        'coords': coords_list,
        'slide_ids': slide_id_list,
        'label_orig_list': label_orig_list,
    }


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
