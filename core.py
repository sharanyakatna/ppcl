"""
PPCL: Plasticity-Preserving Contrastive Learning — core module.

PPCL idea: low-variance projector dims tend to collapse across tasks, so we
designate them as a plasticity reserve and apply a uniformity objective there
to revive spread. High-variance dims do the normal contrastive alignment.
See ppcl_loss docstring for details.

BYOL stability comes from the EMA target; ablation vs explicit L2 anchoring
is in simclr_ema_l2 (see ppcl_theory.py for the weight-space framing).

Run from repo root: `%run core.py` (needs ppcl_subspace.py, ppcl_theory.py).
"""
import os
import sys
import copy
import math
import random
import json
import time
import warnings
import zipfile
import shutil
import threading
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset


def _make_grad_scaler(enabled: bool):
    """Prefer torch.amp.GradScaler; fall back to cuda.amp for older PyTorch."""
    if not torch.cuda.is_available():
        from torch.cuda.amp import GradScaler as _CpuSafe
        return _CpuSafe(enabled=False)
    try:
        from torch.amp import GradScaler as _AmpG  # type: ignore
        return _AmpG("cuda", enabled=enabled)
    except Exception:  # pragma: no cover
        from torch.cuda.amp import GradScaler as _AmpG
        return _AmpG(enabled=enabled)


try:
    from torch.amp import autocast as _autocast  # type: ignore

    def autocast_ctx(enabled):
        return _autocast(device_type="cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _autocast

    def autocast_ctx(enabled):
        return _autocast(enabled=enabled)


from torchvision import datasets, transforms
from tqdm.auto import tqdm
from ppcl_subspace import VarianceEMASubspace, ppcl_loss_stable_subspace
from ppcl_theory import (
    byol_online_target_enc_proj_l2,
    ema_update_modules_,
    simclr_shadow_enc_proj_l2,
    weight_modules_l2_squared_sum,
)

# scipy imported lazily inside sig_test to avoid sympy conflict on Colab
warnings.filterwarnings("ignore")
RESULT_SCHEMA_VERSION = "v2.3"


def _env_truthy(name, default="0"):
    v = os.environ.get(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# --- Paths / Colab (opt-in mount; local-friendly defaults) ---
ON_COLAB = os.environ.get("COLAB_RELEASE_TAG") is not None
SAVE_DIR = os.environ.get("PPCL_SAVE_DIR", "./results")
CKPT_DIR = os.environ.get("PPCL_CKPT_DIR", os.path.join(SAVE_DIR, "checkpoints"))
if ON_COLAB and _env_truthy("PPCL_MOUNT_DRIVE", "1"):
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        SAVE_DIR = "/content/drive/MyDrive/plasticity_collapse"
        CKPT_DIR = "/content/drive/MyDrive/plasticity_collapse/checkpoints"
    except Exception:
        pass
FIG_DIR = os.path.join(SAVE_DIR, "figures")
for d in [SAVE_DIR, CKPT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Device: {DEVICE} | Save: {SAVE_DIR}")


def preflight_check(print_packages=True):
    """Sanity check for a fresh session: imports, device, directories."""
    ok = True
    print("=== PPCL Preflight Check ===")
    print(f"Device: {DEVICE}")
    print(f"SAVE_DIR: {SAVE_DIR}")
    print(f"CKPT_DIR: {CKPT_DIR}")
    for d in [SAVE_DIR, CKPT_DIR, os.path.join(SAVE_DIR, "figures")]:
        if not os.path.isdir(d):
            ok = False
            print(f"[FAIL] Missing directory: {d}")
        else:
            print(f"[OK] Directory exists: {d}")
    if print_packages:
        try:
            import torchvision, platform
            print(f"Python: {platform.python_version()}")
            print(f"Torch: {torch.__version__}")
            print(f"Torchvision: {torchvision.__version__}")
            if torch.cuda.is_available():
                print(f"CUDA: {torch.version.cuda}")
        except Exception as e:
            ok = False
            print(f"[FAIL] Package/version check failed: {e}")
    print("Preflight:", "PASS" if ok else "FAIL")
    return ok


def clear_results_cache(delete_checkpoints=False, delete_figures=False):
    """Delete cached JSON results for a clean rerun."""
    removed = {"json": 0, "pt": 0, "figures": 0}
    for base in [SAVE_DIR, "./results"]:
        if not os.path.isdir(base):
            continue
        for fn in os.listdir(base):
            if fn.endswith(".json"):
                try:
                    os.remove(os.path.join(base, fn))
                    removed["json"] += 1
                except Exception:
                    pass
    if delete_checkpoints:
        for cdir in [CKPT_DIR, "./results/checkpoints"]:
            if not os.path.isdir(cdir):
                continue
            for fn in os.listdir(cdir):
                if fn.endswith(".pt"):
                    try:
                        os.remove(os.path.join(cdir, fn))
                        removed["pt"] += 1
                    except Exception:
                        pass
    if delete_figures:
        for fdir in [os.path.join(SAVE_DIR, "figures"), "./results/figures"]:
            if not os.path.isdir(fdir):
                continue
            for fn in os.listdir(fdir):
                if fn.endswith(".pdf") or fn.endswith(".png") or fn.endswith(".json") or fn.endswith(".txt"):
                    try:
                        os.remove(os.path.join(fdir, fn))
                        removed["figures"] += 1
                    except Exception:
                        pass
    print(f"Cleared cache: json={removed['json']} pt={removed['pt']} figures={removed['figures']}")
    return removed


def _seed_worker(worker_id):
    # Make dataloader workers deterministic too
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader(
    dataset,
    batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
    seed=None,
    persistent_workers=True,
    prefetch_factor=2,
):
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
    use_persistent = persistent_workers and (num_workers is not None and num_workers > 0)
    use_prefetch = use_persistent
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch_factor if use_prefetch else None,
        worker_init_fn=_seed_worker,
        generator=gen,
    )


def set_global_determinism(seed=0, deterministic=False):
    """Seed all RNGs. Pass deterministic=True for strict mode (slower)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _inject_keepalive():
    try:
        import IPython
        IPython.display.display(IPython.display.Javascript('''
        function keepAlive(){
          try{
            var b=document.querySelector("colab-connect-button");
            if(b){b.click();}
            document.dispatchEvent(new Event("mousemove"));
            document.dispatchEvent(new Event("keydown"));
          }catch(e){}
        }
        setInterval(keepAlive,45000);
        document.addEventListener("visibilitychange", ()=>{ if(!document.hidden){ keepAlive(); }});
        '''))
        print("Keep-alive active.")
    except Exception:
        pass


if ON_COLAB and not _env_truthy("PPCL_NO_KEEPALIVE"):
    _inject_keepalive()

_HEARTBEAT_STOP = False


def _write_heartbeat():
    payload = {
        "ts": time.time(),
        "human_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(DEVICE),
        "on_colab": ON_COLAB,
    }
    for base in [SAVE_DIR, "./results"]:
        try:
            os.makedirs(base, exist_ok=True)
            _atomic_json_write(payload, os.path.join(base, "heartbeat.json"))
        except Exception:
            pass


def _start_heartbeat(interval_sec=60):
    def _loop():
        while not _HEARTBEAT_STOP:
            _write_heartbeat()
            time.sleep(interval_sec)
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def _atomic_json_write(obj, path):
    """Write JSON atomically via a temp file to avoid corruption on disconnects."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_crash_log(context, err):
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "context": context,
        "error": str(err),
        "traceback": traceback.format_exc(),
        "device": str(DEVICE),
    }
    for base in [SAVE_DIR, "./results"]:
        try:
            os.makedirs(base, exist_ok=True)
            fn = f"crash_{int(time.time())}.json"
            _atomic_json_write(payload, os.path.join(base, fn))
        except Exception:
            pass


_HB_THREAD = None
if ON_COLAB and not _env_truthy("PPCL_NO_HEARTBEAT"):
    _HB_THREAD = _start_heartbeat(60)
    print("Heartbeat writer active (Drive + local).")

# --- Training hyperparameters ---
EPOCHS = 200
BATCH = 256
LR = 3e-4
WD = 1e-4
TEMP = 0.5
PROJ_DIM = 128
PROJ_HID = 512
ENC_DIM = 512
BYOL_MOM = 0.996
PPCL_RESERVE = 0.10
PPCL_LAM = 0.05
EWC_LAM = 1000.0
LWF_ALPHA = 1.0
LWF_TEMP = 2.0
REPLAY_PER_TASK = 200
KNN_K = 200
# Lambda for simclr_ema_l2 (explicit L2-to-EMA-shadow BYOL ablation)
EXPLICIT_EMA_L2_LAM = 5e-4
# Linear probe eval: SGD + cosine annealing, matching standard SSL protocol
LINEAR_PROBE_EPOCHS = 100
LINEAR_PROBE_LR = 0.1
LINEAR_PROBE_MOMENTUM = 0.9
LINEAR_PROBE_WD = 0.0
# Weight for the BYOL auxiliary term in ppcl_mom
PPCL_MOM_BYOL_AUX_WEIGHT = 0.1
# Cap on Fisher minibatches for EWC (approximation gets good well before full pass)
FISHER_MAX_BATCHES = 500


# --- LARS optimizer ---
class LARS(torch.optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling optimizer.

    From SimCLR / BYOL training setups. Scales the learning rate per layer
    based on the ratio of weight norm to gradient norm. BN params and biases
    are excluded from adaptation, which is the standard practice.

    Reference: You et al., 2017 (large-batch CNNs).
    """

    def __init__(self, params, lr=0.3, momentum=0.9, weight_decay=1e-4,
                 eta=1e-3, exclude_bias_and_bn=True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        eta=eta, exclude_bias_and_bn=exclude_bias_and_bn)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            wd = group['weight_decay']
            eta = group['eta']
            for p in group['params']:
                if p.grad is None:
                    continue
                dp = p.grad
                if wd != 0:
                    dp = dp.add(p, alpha=wd)
                if not group.get('exclude_from_lars', False):
                    p_norm = torch.norm(p)
                    g_norm = torch.norm(dp)
                    if p_norm > 0 and g_norm > 0:
                        local_lr = eta * p_norm / (g_norm + eta * p_norm)
                    else:
                        local_lr = 1.0
                else:
                    local_lr = 1.0
                dp = dp.mul(local_lr * lr)
                if momentum != 0:
                    buf = self.state[p]
                    if 'momentum_buffer' not in buf:
                        buf['momentum_buffer'] = torch.clone(dp).detach()
                    else:
                        buf['momentum_buffer'].mul_(momentum).add_(dp)
                    dp = buf['momentum_buffer']
                p.add_(dp, alpha=-1)
        return loss


def _make_optimizer(model, optimizer_name, lr, method):
    """Build optimizer. Auto-selects LARS for SimCLR/BYOL family."""
    if optimizer_name == "auto":
        optimizer_name = "lars" if method in (
            "simclr", "byol", "simclr_ema_l2", "simclr_pred",
            "simclr_ema_l2_pred", "byol_no_pred",
        ) else "sgd"
    trainable = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "lars":
        # BN and bias params excluded from LARS adaptation (standard)
        lars_params = []
        excluded_params = []
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                for p in m.parameters():
                    if p.requires_grad:
                        excluded_params.append(p)
            elif hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                lars_params.append(m.weight)
                if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                    excluded_params.append(m.bias)
        lars_ids = {id(p) for p in lars_params}
        excl_ids = {id(p) for p in excluded_params}
        for p in trainable:
            if id(p) not in lars_ids and id(p) not in excl_ids:
                excluded_params.append(p)
        param_groups = [
            {'params': lars_params, 'exclude_from_lars': False},
            {'params': excluded_params, 'exclude_from_lars': True},
        ]
        return LARS(param_groups, lr=lr, momentum=0.9, weight_decay=WD)
    else:
        return torch.optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=WD)


CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD  = [0.2023, 0.1994, 0.2010]
TINY_MEAN  = [0.4802, 0.4481, 0.3975]
TINY_STD   = [0.2302, 0.2265, 0.2262]
IN100_MEAN = [0.485, 0.456, 0.406]
IN100_STD  = [0.229, 0.224, 0.225]

RESULT_KEYS = [
    'method', 'avg_acc', 'forgetting', 'backward_transfer', 'fwd_transfer',
    'fwd_transfer_per_task', 'acc_matrix',
    'sranks', 'eranks', 'uniforms', 'lsis', 'ppcl_eranks', 'cka_drifts',
    'grad_aligns', 'plast_ratios', 'time_min', 'seed', 'n_params', 'extra_params',
    # BYOL / explicit-EMA ablation diagnostics
    'byol_target_weight_l2', 'explicit_ema_l2_lambda',
    # Linear probe column (camera-ready)
    'linear_acc', 'linear_accs',
    # Memory usage
    'peak_memory_mb',
    # Versioning
    'schema_version',
]


def _empty_result(method, seed):
    return {k: [] if k in ('acc_matrix', 'sranks', 'eranks', 'uniforms', 'lsis',
        'ppcl_eranks', 'cka_drifts', 'grad_aligns', 'plast_ratios', 'linear_accs',
        'fwd_transfer_per_task', 'byol_target_weight_l2') else
        (method if k == 'method' else seed if k == 'seed' else 0.0 if k in
        ('avg_acc', 'forgetting', 'backward_transfer', 'fwd_transfer', 'time_min',
         'linear_acc', 'explicit_ema_l2_lambda') else ('' if k == 'schema_version' else 0))
        for k in RESULT_KEYS}


# --- Data ---
class TwoViewTransform:
    """Two-view augmentation for SSL (SimCLR/BYOL style).

    RandomResizedCrop, HorizontalFlip, ColorJitter, Grayscale,
    GaussianBlur (kernel ~10% of image size, p=0.5).
    """

    def __init__(self, size=32, mean=None, std=None):
        mean = mean or CIFAR_MEAN
        std = std or CIFAR_STD
        blur_ks = max(3, int(size * 0.1)) | 1  # must be odd
        self.t = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(
                kernel_size=blur_ks, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __call__(self, x):
        return self.t(x), self.t(x)


class SingleViewTransform:
    def __init__(self, size=32, mean=None, std=None):
        mean = mean or CIFAR_MEAN
        std = std or CIFAR_STD
        self.t = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __call__(self, x):
        return self.t(x)


class TaskSplitter:
    def __init__(self, dataset, num_tasks, classes_per_task, seed=42):
        rng = np.random.RandomState(seed)
        targets = np.array(
            dataset.targets if hasattr(dataset, 'targets')
            else [s[1] for s in dataset.samples]
        )
        all_cls = sorted(set(targets.tolist()))
        rng.shuffle(all_cls)
        self.task_classes = [
            all_cls[i * classes_per_task:(i + 1) * classes_per_task]
            for i in range(num_tasks)
        ]
        self.subsets = []
        for cls_list in self.task_classes:
            idx = np.where(np.isin(targets, cls_list))[0].tolist()
            self.subsets.append(Subset(dataset, idx))

    def __len__(self):
        return len(self.subsets)

    def __getitem__(self, i):
        return self.subsets[i]


class ReplayBuffer:
    def __init__(self, per_task=200):
        self.per_task = per_task
        self.images = []
        self.labels = []
        self.stored_reps = []

    def update(self, dataset_subset, ev_tfm, model=None, device=None):
        # Use clean eval-transform images (no augmentation noise in the buffer)
        ds = copy.copy(dataset_subset)
        ds.dataset = copy.copy(dataset_subset.dataset)
        ds.dataset.transform = ev_tfm
        n = len(ds)
        k = min(self.per_task, n)
        idx = np.random.choice(n, k, replace=False)
        imgs = []
        labs = []
        for i in idx:
            img, lab = ds[i]
            if isinstance(img, torch.Tensor):
                imgs.append(img)
            else:
                imgs.append(transforms.ToTensor()(img))
            labs.append(lab)
        self.images.append(torch.stack(imgs))
        self.labels.append(torch.tensor(labs))
        self._cache_valid = False
        self._cache_rep = None
        # Store representations for DER-SSL if model provided
        if model is not None and device is not None:
            model.eval()
            with torch.no_grad():
                batch = self.images[-1].to(device)
                _, reps, _ = model(batch)
                self.stored_reps.append(reps.cpu())
            model.train()
        else:
            self.stored_reps.append(None)

    def _rebuild_cache(self):
        if self.images:
            self._cache_img = torch.cat(self.images)
            self._cache_lab = torch.cat(self.labels)
            valid_reps = [r for r in self.stored_reps if r is not None]
            if len(valid_reps) == len(self.stored_reps) and valid_reps:
                self._cache_rep = torch.cat(valid_reps)
            else:
                self._cache_rep = None
        else:
            self._cache_img = None
            self._cache_lab = None
            self._cache_rep = None
        self._cache_valid = True

    def get_batch(self, n=None):
        if not self.images:
            return None, None
        if not getattr(self, "_cache_valid", False):
            self._rebuild_cache()
        all_img, all_lab = self._cache_img, self._cache_lab
        if n is None or n >= len(all_img):
            return all_img, all_lab
        idx = torch.randperm(len(all_img))[:n]
        return all_img[idx], all_lab[idx]

    def get_batch_with_reps(self, n=None):
        if not self.images:
            return None, None, None
        if not getattr(self, "_cache_valid", False):
            self._rebuild_cache()
        all_img, all_lab = self._cache_img, self._cache_lab
        all_reps = self._cache_rep
        if n is None or n >= len(all_img):
            return all_img, all_lab, all_reps
        idx = torch.randperm(len(all_img))[:n]
        return all_img[idx], all_lab[idx], (all_reps[idx] if all_reps is not None else None)


def _download_tinyimagenet(root="./data"):
    tdir = os.path.join(root, "tiny-imagenet-200")
    val_dir = os.path.join(tdir, "val")
    if os.path.isdir(tdir):
        subdirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
        if len(subdirs) > 5:
            print("TinyImageNet ready.")
            return tdir
    if not os.path.isdir(tdir):
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zpath = os.path.join(root, "tiny-imagenet-200.zip")
        if not os.path.exists(zpath):
            print("Downloading TinyImageNet...")
            import urllib.request
            urllib.request.urlretrieve(url, zpath)
        print("Extracting...")
        with zipfile.ZipFile(zpath, 'r') as z:
            z.extractall(root)
    ann_file = os.path.join(val_dir, "val_annotations.txt")
    if os.path.exists(ann_file):
        print("Restructuring val directory...")
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                fname, cls = parts[0], parts[1]
                cls_dir = os.path.join(val_dir, cls)
                os.makedirs(cls_dir, exist_ok=True)
                src = os.path.join(val_dir, "images", fname)
                dst = os.path.join(cls_dir, fname)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.move(src, dst)
        img_dir = os.path.join(val_dir, "images")
        if os.path.isdir(img_dir) and not os.listdir(img_dir):
            os.rmdir(img_dir)
    print("TinyImageNet ready.")
    return tdir


def _setup_imagenet100(root="./data/imagenet100"):
    """Check ImageNet-100 is present at root.
    Expected layout: {root}/train/<class>/, {root}/val/<class>/
    """
    train_dir = os.path.join(root, "train")
    if os.path.isdir(train_dir):
        n_classes = len([d for d in os.listdir(train_dir)
                         if os.path.isdir(os.path.join(train_dir, d))])
        if n_classes >= 100:
            print(f"ImageNet-100 ready ({n_classes} classes).")
            return root
    raise FileNotFoundError(
        f"ImageNet-100 not found at {root}. "
        "Set PPCL_IN100_DIR or IMAGENET100_ROOT, or download from Kaggle (imagenet-100). "
        "Expected: {root}/train/<class_dir>/, {root}/val/<class_dir>/"
    )


def get_task_loaders(dataset_name="cifar100", num_tasks=10, classes_per_task=10,
                     batch_size=256, seed=42):
    if dataset_name == "cifar100":
        tr_ds = datasets.CIFAR100("./data", train=True, download=True)
        te_ds = datasets.CIFAR100("./data", train=False, download=True)
        ssl_tfm = TwoViewTransform(32, CIFAR_MEAN, CIFAR_STD)
        ev_tfm = SingleViewTransform(32, CIFAR_MEAN, CIFAR_STD)
    elif dataset_name == "tinyimagenet":
        tdir = _download_tinyimagenet()
        tr_ds = datasets.ImageFolder(os.path.join(tdir, "train"))
        te_ds = datasets.ImageFolder(os.path.join(tdir, "val"))
        ssl_tfm = TwoViewTransform(64, TINY_MEAN, TINY_STD)
        ev_tfm = SingleViewTransform(64, TINY_MEAN, TINY_STD)
    elif dataset_name == "imagenet100":
        root = (
            os.environ.get("IMAGENET100_ROOT")
            or os.environ.get("PPCL_IN100_DIR")
            or "./data/imagenet100"
        )
        root = root.rstrip("/")
        _setup_imagenet100(root)
        tr_ds = datasets.ImageFolder(os.path.join(root, "train"))
        te_ds = datasets.ImageFolder(os.path.join(root, "val"))
        ssl_tfm = TwoViewTransform(224, IN100_MEAN, IN100_STD)
        ev_tfm = SingleViewTransform(224, IN100_MEAN, IN100_STD)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    sp_tr = TaskSplitter(tr_ds, num_tasks, classes_per_task, seed)
    sp_te = TaskSplitter(te_ds, num_tasks, classes_per_task, seed)
    n_classes = num_tasks * classes_per_task
    return sp_tr, sp_te, ssl_tfm, ev_tfm, n_classes


# --- Models ---
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or inp != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18Small(nn.Module):
    """CIFAR-adapted ResNet-18.

    Uses a 3x3 stem (no 7x7), no max-pool, stride-1 first layer for inputs <=64px.
    Matches the CIFAR variant used in Solo-Learn, CaSSLe, PNR, etc.
    Output: 512-d after adaptive average pooling.
    """

    def __init__(self, in_size=32):
        super().__init__()
        self.in_planes = 64
        s = 1 if in_size <= 64 else 2
        self.conv1 = nn.Conv2d(3, 64, 3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._layer(64, 2, 1)
        self.layer2 = self._layer(128, 2, 2)
        self.layer3 = self._layer(256, 2, 2)
        self.layer4 = self._layer(512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _layer(self, planes, n, stride):
        strides = [stride] + [1] * (n - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.pool(x).flatten(1)


class ProjectionHead(nn.Module):
    def __init__(self, ind=512, hid=512, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ind, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(True),
            nn.Linear(hid, out),
        )

    def forward(self, x):
        return self.net(x)


class PredictorHead(nn.Module):
    def __init__(self, ind=128, hid=512, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ind, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(True),
            nn.Linear(hid, out),
        )

    def forward(self, x):
        return self.net(x)


class SSLModel(nn.Module):
    def __init__(self, use_pred=False, use_ema=False, ema_mom=0.996,
                 in_size=32, proj_dim=128, proj_hid=512):
        super().__init__()
        if in_size >= 224:
            import torchvision.models as tvm
            _r18 = tvm.resnet18(weights=None)
            # Standard ImageNet ResNet-18 with 7x7 stem for 224x224 inputs
            self.enc = nn.Sequential(*list(_r18.children())[:-1], nn.Flatten())
        else:
            self.enc = ResNet18Small(in_size)
        enc_dim = 512
        self.proj = ProjectionHead(ind=enc_dim, hid=proj_hid, out=proj_dim)
        self.use_pred = use_pred
        self.use_ema = use_ema
        self.ema_mom = ema_mom
        if use_pred:
            self.pred = PredictorHead(ind=proj_dim, hid=proj_hid, out=proj_dim)
        if use_ema:
            self.t_enc = copy.deepcopy(self.enc)
            self.t_proj = copy.deepcopy(self.proj)
            for p in list(self.t_enc.parameters()) + list(self.t_proj.parameters()):
                p.requires_grad = False

    def forward(self, x):
        h = self.enc(x)
        z = self.proj(h)
        p = self.pred(z) if self.use_pred else z
        return h, z, p

    def forward_target(self, x):
        with torch.no_grad():
            h = self.t_enc(x)
            z = self.t_proj(h)
        return z

    @torch.no_grad()
    def update_ema(self):
        if not self.use_ema:
            return
        tau = self.ema_mom
        for op, tp in zip(self.enc.parameters(), self.t_enc.parameters()):
            tp.data.mul_(tau).add_(op.data, alpha=1 - tau)
        for op, tp in zip(self.proj.parameters(), self.t_proj.parameters()):
            tp.data.mul_(tau).add_(op.data, alpha=1 - tau)

    def reinit_projector(self):
        for m in self.proj.modules():
            if isinstance(m, (nn.Linear, nn.BatchNorm1d)):
                m.reset_parameters()
        if self.use_pred:
            for m in self.pred.modules():
                if isinstance(m, (nn.Linear, nn.BatchNorm1d)):
                    m.reset_parameters()


# --- Losses ---
def ntxent(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.5) -> torch.Tensor:
    """NT-Xent (SimCLR) loss.

    Must run outside autocast: fp16 logits + masked_fill(-1e9) overflows.
    """
    with autocast_ctx(False):
        z1 = F.normalize(z1.float(), dim=1)
        z2 = F.normalize(z2.float(), dim=1)
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        logits = (z @ z.t()) / temp      # float32
        # Out-of-place mask so we don't write to tensors that may be in autograd graph
        mask = torch.eye(2 * B, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(mask, -1e9)
        targets = torch.cat(
            [
                torch.arange(B, device=logits.device) + B,
                torch.arange(B, device=logits.device),
            ],
            dim=0,
        )
        return F.cross_entropy(logits, targets, reduction="mean")


def ppcl_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    reserve: float = 0.1,
    temp: float = 0.5,
    unif_weight: float = 0.05,
) -> Tuple[torch.Tensor, float, float, float]:
    """Plasticity-Preserving Contrastive Loss.

    Low-variance dims (collapse-prone) = plasticity reserve.
    Uniformity loss revives spread there (Wang & Isola 2020 Gaussian kernel).
    High-variance dims = active subspace, carrying the NT-Xent objective.

    reserve: fraction of dims assigned to the plasticity reserve.
    Dim selection is stop-grad (discrete topk on variance); gradients only
    flow through the NT-Xent and uniformity terms.
    """
    z1, z2 = z1.float(), z2.float()  # float16 overflows under AMP
    D = z1.shape[1]
    num_r = max(1, int(D * reserve))
    with torch.no_grad():
        var = torch.cat([z1, z2], 0).var(dim=0)
        _, low_idx = var.topk(num_r, largest=False)
        hmask = torch.ones(D, dtype=torch.bool, device=z1.device)
        hmask[low_idx] = False
        high_idx = torch.where(hmask)[0]
    Lc = ntxent(z1[:, high_idx], z2[:, high_idx], temp)
    zf = F.normalize(torch.cat([z1[:, low_idx], z2[:, low_idx]], 0).float(), dim=1)
    # O(N^2) in batch; fine for B<=512
    Lu = torch.pdist(zf, p=2).pow(2).mul(-2).exp().mean().clamp(min=1e-8).log()
    er = 0.0  # skip SVD on hot path; saves ~40% of PPCL training time
    return Lc + unif_weight * Lu, er, Lc.item(), Lu.item()


def ppcl_loss_fixed_low(
    z1: torch.Tensor,
    z2: torch.Tensor,
    low_idx: Union[torch.Tensor, List[int]],
    temp: float = 0.5,
    unif_weight: float = 0.05,
) -> Tuple[torch.Tensor, float, float, float]:
    """PPCL with fixed (pre-computed) plasticity dimensions.

    Used for the random-dims ablation so the reserve is held constant
    across the run rather than re-selected each batch.
    """
    z1, z2 = z1.float(), z2.float()
    D = z1.shape[1]
    if not torch.is_tensor(low_idx):
        low_idx = torch.tensor(low_idx, dtype=torch.long)
    low_idx = low_idx.to(z1.device)

    hmask = torch.ones(D, dtype=torch.bool, device=z1.device)
    hmask[low_idx] = False
    high_idx = torch.where(hmask)[0]

    Lc = ntxent(z1[:, high_idx], z2[:, high_idx], temp)
    zf = F.normalize(torch.cat([z1[:, low_idx], z2[:, low_idx]], 0).float(), dim=1)
    Lu = torch.pdist(zf, p=2).pow(2).mul(-2).exp().mean().clamp(min=1e-8).log()
    return Lc + unif_weight * Lu, 0.0, Lc.item(), Lu.item()


def ppcl_loss_soft(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temp: float = 0.5,
    unif_weight: float = 0.05,
) -> Tuple[torch.Tensor, float, float, float]:
    """Soft-PPCL: continuous per-dimension weighting instead of a hard partition.

    Weight per dim: w_d = exp(-var_d / median_var).
    High weight on low-variance (collapsing) dims, low on healthy ones.
    Strictly generalizes hard PPCL (binary w is the limit).
    """
    z1f, z2f = z1.float(), z2.float()
    D = z1f.shape[1]

    Lc = ntxent(z1f, z2f, temp)

    with torch.no_grad():
        var = torch.cat([z1f, z2f], 0).var(dim=0)  # [D]
        median_var = var.median().clamp(min=1e-8)
        w = torch.exp(-var / median_var)

    zcat = torch.cat([z1f, z2f], 0)  # [2B, D]
    zw = zcat * w.unsqueeze(0).sqrt()
    zw_norm = F.normalize(zw, dim=1)
    Lu = torch.pdist(zw_norm, p=2).pow(2).mul(-2).exp().mean().clamp(min=1e-8).log()

    loss = Lc + unif_weight * Lu
    return loss, 0.0, Lc.item(), Lu.item()


def byol_loss(
    p1: torch.Tensor,
    z2t: torch.Tensor,
    p2: torch.Tensor,
    z1t: torch.Tensor,
) -> torch.Tensor:
    """BYOL cosine loss: predictor vs stop-grad EMA target projections."""
    p1, p2 = F.normalize(p1, dim=1), F.normalize(p2, dim=1)
    z1t, z2t = F.normalize(z1t.detach(), dim=1), F.normalize(z2t.detach(), dim=1)
    return 2 - (p1 * z2t).sum(1).mean() - (p2 * z1t).sum(1).mean()


def barlow_loss(z1: torch.Tensor, z2: torch.Tensor, lam: float = 5e-3) -> torch.Tensor:
    if z1.size(0) < 2:
        raise ValueError("barlow_loss requires batch size >= 2 for cross-correlation.")
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-5)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-5)
    C = z1.T @ z2 / z1.size(0)
    on = (1 - C.diagonal()).pow(2).sum()
    C2 = C.pow(2)
    off = (C2 - torch.diag(C2.diag())).sum()
    return on + lam * off


def simsiam_loss(
    p1: torch.Tensor,
    z2: torch.Tensor,
    p2: torch.Tensor,
    z1: torch.Tensor,
) -> torch.Tensor:
    """SimSiam: -0.5 * (cos(p1, z2) + cos(p2, z1)), targets stop-grad."""
    z1, z2 = z1.detach(), z2.detach()
    return -0.5 * (
        (F.normalize(p1, dim=1) * F.normalize(z2, dim=1)).sum(1).mean()
        + (F.normalize(p2, dim=1) * F.normalize(z1, dim=1)).sum(1).mean()
    )


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_w: float = 25,
    var_w: float = 25,
    cov_w: float = 1,
) -> torch.Tensor:
    if z1.size(0) < 2:
        raise ValueError("vicreg_loss requires batch size >= 2 for covariance (uses B-1).")
    sim = F.mse_loss(z1, z2)
    z1c, z2c = z1 - z1.mean(0), z2 - z2.mean(0)
    std1, std2 = z1c.std(0), z2c.std(0)
    var = (F.relu(1 - std1).mean() + F.relu(1 - std2).mean()) / 2
    B, D = z1.size()
    cov1 = (z1c.T @ z1c) / (B - 1)
    cov2 = (z2c.T @ z2c) / (B - 1)
    c1sq = cov1.pow(2)
    c2sq = cov2.pow(2)
    cov = ((c1sq - torch.diag(c1sq.diag())).sum() / D
           + (c2sq - torch.diag(c2sq.diag())).sum() / D) / 2
    return sim_w * sim + var_w * var + cov_w * cov


def distill_loss(z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    z_s = F.log_softmax(z_s / 2, dim=1)
    z_t = F.softmax(z_t.detach() / 2, dim=1)
    return F.kl_div(z_s, z_t, reduction='batchmean') * (2 ** 2)


def lump_mixup(
    v1: torch.Tensor,
    v2: torch.Tensor,
    buf_imgs: Optional[torch.Tensor],
    ssl_aug: Any,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if buf_imgs is None:
        return v1, v2
    lam = np.random.beta(alpha, alpha)
    n = min(v1.size(0), buf_imgs.size(0))
    idx = torch.randperm(buf_imgs.size(0))[:n]
    # Clone to avoid in-place modification — can break autograd / torch.compile
    v1 = v1.clone()
    v1[:n] = lam * v1[:n] + (1 - lam) * buf_imgs[idx].to(v1.device)
    return v1, v2


# --- Evaluation ---
def _train_linear_probe_sgd(clf, tr_f, tr_y, device):
    """Frozen-encoder linear probe: SGD + cosine LR (standard SSL eval protocol)."""
    opt = torch.optim.SGD(
        clf.parameters(),
        lr=LINEAR_PROBE_LR,
        momentum=LINEAR_PROBE_MOMENTUM,
        weight_decay=LINEAR_PROBE_WD,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, LINEAR_PROBE_EPOCHS))
    ds = TensorDataset(tr_f, tr_y)
    dl = _make_loader(ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True, seed=0)
    clf.train()
    for _ in range(LINEAR_PROBE_EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(clf(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()


@torch.no_grad()
def _extract_batched(model, subset, ev_tfm, device, bs=256):
    """Batched feature extraction. Creates a fresh DataLoader each call."""
    ds_copy = copy.copy(subset)
    ds_copy.dataset = copy.copy(subset.dataset)
    ds_copy.dataset.transform = ev_tfm
    dl = _make_loader(ds_copy, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    feats, labs = [], []
    model.eval()
    for xb, yb in dl:
        xb = xb.to(device)
        h = model.enc(xb)
        feats.append(h.cpu())
        labs.append(yb)
    return torch.cat(feats), torch.cat(labs)


def knn_eval(
    model: nn.Module,
    train_sub,
    test_sub,
    ev_tfm,
    k: int = 200,
    device: Optional[torch.device] = None,
) -> float:
    device = device or DEVICE
    tr_f, tr_y = _extract_batched(model, train_sub, ev_tfm, device)
    te_f, te_y = _extract_batched(model, test_sub, ev_tfm, device)
    tr_f = F.normalize(tr_f, dim=1)
    te_f = F.normalize(te_f, dim=1)
    n_tr = tr_f.size(0)
    k_eff = min(k, max(1, n_tr // 10))
    correct = total = 0
    for i in range(0, te_f.size(0), 256):
        sim = te_f[i:i + 256] @ tr_f.t()
        _, topk_idx = sim.topk(min(k_eff, n_tr), dim=1)
        pred = tr_y[topk_idx].mode(dim=1).values
        correct += (pred == te_y[i:i + 256]).sum().item()
        total += min(256, te_f.size(0) - i)
    return correct / total * 100


def linear_probe_eval(
    model: nn.Module,
    sp_tr,
    sp_te,
    n_tasks: int,
    ev_tfm,
    n_classes: int,
    device: Optional[torch.device] = None,
) -> Tuple[float, List[float]]:
    device = device or DEVICE
    model.eval()
    all_tr_f, all_tr_y, all_te_f, all_te_y = [], [], [], []
    for t in range(n_tasks):
        f, y = _extract_batched(model, sp_tr[t], ev_tfm, device)
        all_tr_f.append(f)
        all_tr_y.append(y)
        f, y = _extract_batched(model, sp_te[t], ev_tfm, device)
        all_te_f.append(f)
        all_te_y.append(y)
    tr_f, tr_y = torch.cat(all_tr_f), torch.cat(all_tr_y)
    te_f, te_y = torch.cat(all_te_f), torch.cat(all_te_y)
    clf = nn.Linear(tr_f.size(1), n_classes).to(device)
    _train_linear_probe_sgd(clf, tr_f, tr_y, device)
    clf.eval()
    with torch.no_grad():
        pred = clf(te_f.to(device)).argmax(1).cpu()
    acc = (pred == te_y).float().mean().item() * 100
    per_task = []
    offset = 0
    for t in range(n_tasks):
        n = all_te_y[t].size(0)
        pt = (pred[offset:offset + n] == te_y[offset:offset + n]).float().mean().item() * 100
        per_task.append(pt)
        offset += n
    return acc, per_task


def transfer_eval(model, target, device=None):
    device = device or DEVICE
    model.eval()
    if target == "cifar10":
        tr = datasets.CIFAR10("./data", train=True, download=True,
                               transform=SingleViewTransform(32))
        te = datasets.CIFAR10("./data", train=False, download=True,
                               transform=SingleViewTransform(32))
        nc = 10
    elif target == "stl10":
        tr = datasets.STL10("./data", split="train", download=True,
                            transform=SingleViewTransform(32, CIFAR_MEAN, CIFAR_STD))
        te = datasets.STL10("./data", split="test", download=True,
                            transform=SingleViewTransform(32, CIFAR_MEAN, CIFAR_STD))
        nc = 10
    else:
        raise ValueError(target)
    tr_dl = _make_loader(tr, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    te_dl = _make_loader(te, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    tr_f, tr_y, te_f, te_y = [], [], [], []
    with torch.no_grad():
        for xb, yb in tr_dl:
            tr_f.append(model.enc(xb.to(device)).cpu())
            tr_y.append(yb)
        for xb, yb in te_dl:
            te_f.append(model.enc(xb.to(device)).cpu())
            te_y.append(yb)
    tr_f, tr_y = torch.cat(tr_f), torch.cat(tr_y)
    te_f, te_y = torch.cat(te_f), torch.cat(te_y)
    clf = nn.Linear(tr_f.size(1), nc).to(device)
    _train_linear_probe_sgd(clf, tr_f, tr_y, device)
    clf.eval()
    with torch.no_grad():
        pred = clf(te_f.to(device)).argmax(1).cpu()
    return (pred == te_y).float().mean().item() * 100


# --- Metrics ---
@torch.no_grad()
def compute_stable_rank(model: nn.Module) -> float:
    ranks = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            W = m.weight.data.float().view(m.weight.size(0), -1)
            s = torch.linalg.svdvals(W)
            ranks.append((s.sum() ** 2 / (s ** 2).sum()).item())
    return np.mean(ranks) if ranks else 0.0


@torch.no_grad()
def compute_erank(
    model: nn.Module,
    subset,
    ev_tfm,
    device: Optional[torch.device] = None,
) -> float:
    device = device or DEVICE
    f, _ = _extract_batched(model, subset, ev_tfm, device)
    f = F.normalize(f, dim=1)
    s = torch.linalg.svdvals(f.float())
    p = s / s.sum()
    p = p[p > 1e-10]
    return torch.exp(-(p * p.log()).sum()).item()


@torch.no_grad()
def compute_uniformity(
    model: nn.Module,
    subset,
    ev_tfm,
    device: Optional[torch.device] = None,
) -> float:
    device = device or DEVICE
    f, _ = _extract_batched(model, subset, ev_tfm, device)
    f = F.normalize(f, dim=1)
    return torch.pdist(f, p=2).pow(2).mul(-2).exp().mean().clamp(min=1e-8).log().item()


@torch.no_grad()
def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA (Kornblith et al., ICML 2019)."""
    X, Y = X - X.mean(0), Y - Y.mean(0)
    K = X @ X.t()
    L = Y @ Y.t()
    hsic_xy = (K * L).sum()
    hsic_xx = (K * K).sum()
    hsic_yy = (L * L).sum()
    return (hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt() + 1e-10)).item()


@torch.no_grad()
def compute_lsi(
    model: nn.Module,
    subset,
    ev_tfm,
    device: Optional[torch.device] = None,
) -> float:
    device = device or DEVICE
    f, y = _extract_batched(model, subset, ev_tfm, device)
    f = F.normalize(f, dim=1)
    classes = y.unique()
    if len(classes) < 2:
        return 0.0
    centroids = torch.stack([f[y == c].mean(0) for c in classes])
    inter = torch.pdist(centroids).mean().item()
    intra = np.mean([
        torch.pdist(f[y == c]).mean().item()
        for c in classes if (y == c).sum() > 1
    ])
    return inter / (intra + 1e-10)


def compute_forgetting(acc_matrix: List[List[float]]) -> float:
    if len(acc_matrix) < 2:
        return 0.0
    T = len(acc_matrix)
    fgt = 0
    for j in range(T - 1):
        candidates = [acc_matrix[k][j] for k in range(j, T) if j < len(acc_matrix[k])]
        if not candidates or j >= len(acc_matrix[-1]):
            continue
        best = max(candidates)
        fgt += max(0, best - acc_matrix[-1][j])
    return fgt / (T - 1)


def compute_backward_transfer(acc_matrix: List[List[float]]) -> float:
    """BWT: (1/(T-1)) sum_j (final acc on task j - acc right after task j)."""
    if len(acc_matrix) < 2:
        return 0.0
    T = len(acc_matrix)
    s = 0.0
    cnt = 0
    for j in range(T - 1):
        if j >= len(acc_matrix[j]) or j >= len(acc_matrix[-1]):
            continue
        s += acc_matrix[-1][j] - acc_matrix[j][j]
        cnt += 1
    return s / max(cnt, 1)


def compute_fwd_transfer_mean(prenotrain_task_accs: List[float]) -> float:
    """Mean kNN accuracy on task t before training on task t (t>=1).

    Note: the old acc_matrix[t-1][t] indexing was OOB. This takes a flat
    list of pre-train accuracies collected during the task loop instead.
    """
    if not prenotrain_task_accs:
        return 0.0
    return float(sum(prenotrain_task_accs) / len(prenotrain_task_accs))


def compute_fwd_transfer(prenotrain_or_legacy):
    """Same as compute_fwd_transfer_mean for a flat list.

    Returns 0.0 for legacy acc_matrix inputs (FWT is undefined there).
    """
    if not prenotrain_or_legacy:
        return 0.0
    if isinstance(prenotrain_or_legacy[0], (list, tuple)):
        return 0.0
    return compute_fwd_transfer_mean(prenotrain_or_legacy)


def sig_test(a: Sequence[float], b: Sequence[float]) -> str:
    """Paired Wilcoxon signed-rank test. Returns significance stars or 'n/a'."""
    a, b = np.array(a), np.array(b)
    if len(a) < 3:
        return "n/a"
    try:
        from scipy import stats as _stats
        _, pw = _stats.wilcoxon(a, b)
        if pw < 0.001:
            return "***"
        if pw < 0.01:
            return "**"
        if pw < 0.05:
            return "*"
        return "ns"
    except Exception:
        return "n/a"


def paired_stats(a, b):
    """Paired comparison stats for paper tables. Wilcoxon only (no cherry-picking)."""
    a, b = np.array(a), np.array(b)
    out = {
        "n": int(min(len(a), len(b))),
        "mean_a": float(np.mean(a)) if len(a) else 0.0,
        "mean_b": float(np.mean(b)) if len(b) else 0.0,
        "delta": float(np.mean(a) - np.mean(b)) if len(a) and len(b) else 0.0,
        "p_wilcoxon": None,
        "p_ttest": None,
        "sig": "n/a",
    }
    if out["n"] < 3:
        return out
    try:
        from scipy import stats as _stats
        _, pw = _stats.wilcoxon(a, b)
        _, pt = _stats.ttest_rel(a, b)
        out["p_wilcoxon"] = float(pw)
        out["p_ttest"] = float(pt)
        if pw < 0.001:
            out["sig"] = "***"
        elif pw < 0.01:
            out["sig"] = "**"
        elif pw < 0.05:
            out["sig"] = "*"
        else:
            out["sig"] = "ns"
    except Exception:
        pass
    return out


# --- Save / load ---
def save_result(result, tag):
    """Save to Drive (primary) and ./results (backup). Never drops data on Drive failure."""
    fname = f"{result['method']}_seed{result['seed']}_{tag}.json"

    def _conv(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, list): return [_conv(x) for x in v]
        return v

    out = {k: _conv(v) for k, v in result.items()}
    saved = []
    for attempt in range(2):
        try:
            path = os.path.join(SAVE_DIR, fname)
            _atomic_json_write(out, path)
            saved.append("Drive")
            break
        except Exception as e:
            if attempt == 1:
                print(f"  Drive save failed: {e}")
            time.sleep(1.0)
    local_dir = "./results"
    os.makedirs(local_dir, exist_ok=True)
    try:
        _atomic_json_write(out, os.path.join(local_dir, fname))
        saved.append("local")
    except Exception as e:
        print(f"  Local save failed: {e}")
    if saved:
        print(f"  Saved to {' + '.join(saved)}: {fname}")
    else:
        print(f"  ALL SAVES FAILED for {fname} — result may be lost!")


def load_result(method, seed, tag):
    path = os.path.join(SAVE_DIR, f"{method}_seed{seed}_{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_all_results(pattern):
    import glob
    results = {}
    for fp in glob.glob(os.path.join(SAVE_DIR, pattern)):
        with open(fp) as f:
            r = json.load(f)
            results[os.path.basename(fp).replace('.json', '')] = r
    return results


def _latest_partial_result(method, seed, ckpt_tag):
    """Find the most recent per-task partial JSON for resuming interrupted runs."""
    import glob
    import re
    pat = re.compile(
        rf"^{re.escape(method)}_seed{int(seed)}_{re.escape(ckpt_tag)}_partial_t(\d+)\.json$"
    )
    best_t = -1
    best = None
    for base in [SAVE_DIR, "./results"]:
        for fp in glob.glob(os.path.join(base, f"{method}_seed{seed}_{ckpt_tag}_partial_t*.json")):
            m = pat.match(os.path.basename(fp))
            if not m:
                continue
            t = int(m.group(1))
            if t < best_t:
                continue
            try:
                with open(fp) as f:
                    r = json.load(f)
                best_t = t
                best = r
            except Exception:
                continue
    return best_t, best


def _atomic_checkpoint_save(state_dict, path):
    """Write .pt atomically to avoid corruption on disconnects."""
    tmp = f"{path}.tmp"
    torch.save(state_dict, tmp)
    os.replace(tmp, path)


def save_checkpoint(model, method, seed, tag):
    """Save model checkpoint to Drive and local backup (both atomic)."""
    fname = f"{method}_seed{seed}_{tag}.pt"
    saved = []
    sd = model.state_dict()
    for attempt in range(2):
        try:
            path = os.path.join(CKPT_DIR, fname)
            _atomic_checkpoint_save(sd, path)
            saved.append("Drive")
            break
        except Exception as e:
            if attempt == 1:
                print(f"  Drive checkpoint save failed: {e}")
            time.sleep(1.0)
    local_dir = "./results/checkpoints"
    os.makedirs(local_dir, exist_ok=True)
    try:
        local_path = os.path.join(local_dir, fname)
        _atomic_checkpoint_save(sd, local_path)
        saved.append("local")
    except Exception as e:
        print(f"  Local checkpoint save failed: {e}")
    if saved:
        print(f"  Checkpoint saved to {' + '.join(saved)}: {fname}")


def load_checkpoint(model, method, seed, tag):
    path = os.path.join(CKPT_DIR, f"{method}_seed{seed}_{tag}.pt")
    if not os.path.exists(path):
        return False
    try:
        sd = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        # weights_only not available in PyTorch < 2.0
        sd = torch.load(path, map_location=DEVICE)
    model.load_state_dict(sd)
    return True


# --- Main trainer ---
def run_method(
    method: str,
    dataset_name: str = "cifar100",
    n_tasks: int = 10,
    epochs: int = 200,
    seed: int = 0,
    task_seed: Optional[int] = None,
    ppcl_reserve: float = 0.1,
    byol_mom: float = 0.996,
    batch_size: int = 256,
    lr: float = 3e-4,
    proj_dim: int = 128,
    buffer_size: int = 200,
    save_ckpt: bool = False,
    load_ckpt: bool = False,
    ckpt_tag: str = "",
    device: Optional[torch.device] = None,
    verbose: bool = True,
    deterministic: bool = False,
    autosave_every_task: bool = True,
    save_ckpt_each_task: bool = False,
    eval_linear_probe: bool = True,
    warmup_epochs: int = 10,
    optimizer: str = "auto",
    ppcl_ema_momentum: float = 0.99,
    ppcl_orth_lambda: float = 0.02,
    explicit_ema_l2_lambda: Optional[float] = None,
    ppcl_lam: float = 0.05,
    ppcl_warmup_tasks: int = 0,
    ppcl_lam_ramp_epochs: int = 10,
) -> Tuple[Dict[str, Any], nn.Module]:
    device = device or DEVICE
    if explicit_ema_l2_lambda is None:
        explicit_ema_l2_lambda = EXPLICIT_EMA_L2_LAM
    explicit_ema_l2_lambda = float(explicit_ema_l2_lambda)
    set_global_determinism(seed, deterministic=deterministic)
    t0 = time.time()
    use_amp = (str(device) == "cuda")
    scaler = _make_grad_scaler(use_amp)
    _total_classes = {"cifar100": 100, "tinyimagenet": 200, "imagenet100": 100}
    nc_per_t = _total_classes.get(dataset_name, 100) // n_tasks
    if task_seed is None:
        task_seed = seed
    sp_tr, sp_te, ssl_tfm, ev_tfm, nc = get_task_loaders(
        dataset_name, n_tasks, nc_per_t, batch_size, task_seed
    )
    in_size = {"cifar100": 32, "tinyimagenet": 64, "imagenet100": 224}.get(dataset_name, 32)
    is_ema = method in ("byol", "ppcl_mom", "byol_no_pred")
    is_pred = method in ("byol", "ppcl_mom", "simsiam", "simclr_pred", "simclr_ema_l2_pred")
    proj_dim_eff = 256 if dataset_name == "imagenet100" else proj_dim
    proj_hid_eff = 2048 if dataset_name == "imagenet100" else PROJ_HID
    model = SSLModel(
        use_pred=is_pred, use_ema=is_ema, ema_mom=byol_mom,
        in_size=in_size, proj_dim=proj_dim_eff, proj_hid=proj_hid_eff,
    ).to(device)
    if load_ckpt and ckpt_tag:
        if load_checkpoint(model, method, seed, ckpt_tag):
            if verbose:
                print(f"  Loaded checkpoint {method} seed={seed} tag={ckpt_tag}")
    n_params = sum(p.numel() for p in model.parameters())
    if method in ("simclr_ema_l2", "simclr_ema_l2_pred"):
        enc_p = sum(p.numel() for p in model.enc.parameters())
        proj_p = sum(p.numel() for p in model.proj.parameters())
        extra_params = enc_p + proj_p
    elif is_ema:
        extra_params = (
            sum(p.numel() for p in model.t_enc.parameters())
            + sum(p.numel() for p in model.t_proj.parameters())
        )
    else:
        extra_params = 0

    # Oracle: train on all tasks jointly
    if method == "oracle":
        return _run_oracle(
            model, sp_tr, sp_te, n_tasks, ssl_tfm, ev_tfm, nc,
            epochs, seed, device, use_amp, scaler, n_params, t0, save_ckpt, ckpt_tag,
        )
    # Supervised baseline
    if method == "supervised":
        return _run_supervised(
            model, sp_tr, sp_te, n_tasks, ev_tfm, nc,
            epochs, seed, device, use_amp, scaler, n_params, t0,
        )

    # Continual learning loop
    replay_buf = ReplayBuffer(buffer_size)
    prev_model = None
    fisher = None
    prev_params = None
    acc_matrix = []
    sranks = []
    eranks = []
    uniforms = []
    lsis = []
    ppcl_eranks = []
    cka_drifts = []
    grad_aligns = []
    plast_ratios = []
    feat_snap = None
    rand_low_idx = None
    fwd_transfer_scores = []
    byol_target_l2_trace = []
    shadow_enc, shadow_proj = None, None
    start_task = 0
    resume_ok_methods = {
        "simclr", "simclr_pred", "simclr_ema_l2", "simclr_ema_l2_pred",
        "ppcl", "ppcl_stable", "ppcl_soft", "ppcl_enc", "ppcl_adaptive",
        "ppcl_mom", "ppcl_rand", "byol", "byol_no_pred", "barlow",
        "vicreg", "simsiam", "pnr",
    }
    run_tag = ckpt_tag or "run"
    if load_ckpt and run_tag and method in resume_ok_methods:
        last_t, last_partial = _latest_partial_result(method, seed, run_tag)
        if last_partial is not None and last_t >= 0:
            if load_checkpoint(model, method, seed, f"{run_tag}_task{last_t}"):
                start_task = int(last_t) + 1
                acc_matrix = list(last_partial.get("acc_matrix", []))
                sranks = list(last_partial.get("sranks", []))
                eranks = list(last_partial.get("eranks", []))
                uniforms = list(last_partial.get("uniforms", []))
                lsis = list(last_partial.get("lsis", []))
                ppcl_eranks = list(last_partial.get("ppcl_eranks", []))
                cka_drifts = list(last_partial.get("cka_drifts", []))
                grad_aligns = list(last_partial.get("grad_aligns", []))
                plast_ratios = list(last_partial.get("plast_ratios", []))
                fwd_transfer_scores = list(last_partial.get("fwd_transfer_per_task", []))
                byol_target_l2_trace = list(last_partial.get("byol_target_weight_l2", []))
                if verbose:
                    print(f"  Resuming {method} seed={seed} from task {start_task}/{n_tasks-1} [{run_tag}]")
            elif verbose:
                print(f"  Partial found but checkpoint missing for task {last_t}; restarting from task 0.")

    # Fixed random dims for ppcl_rand (held constant across the whole run)
    if method == "ppcl_rand":
        D = proj_dim
        num_r = max(1, int(D * ppcl_reserve))
        g = torch.Generator()
        g.manual_seed(int(seed))
        perm = torch.randperm(D, generator=g)
        rand_low_idx = perm[:num_r].clone()

    for t in range(start_task, n_tasks):
        tr_sub = sp_tr[t]
        te_sub = sp_te[t]
        # Forward transfer: eval on next task before any training on it
        if t > 0:
            model.eval()
            fwd_transfer_scores.append(
                knn_eval(model, sp_tr[t], sp_te[t], ev_tfm, KNN_K, device)
            )
        stable_sub = None
        if method == "ppcl_stable":
            stable_sub = VarianceEMASubspace(proj_dim_eff, momentum=ppcl_ema_momentum)
        if method in ("simclr_ema_l2", "simclr_ema_l2_pred"):
            shadow_enc = copy.deepcopy(model.enc).to(device)
            shadow_proj = copy.deepcopy(model.proj).to(device)
            for sm in (shadow_enc, shadow_proj):
                sm.eval()
                for q in sm.parameters():
                    q.requires_grad_(False)
        tr_ds = copy.copy(tr_sub)
        tr_ds.dataset = copy.copy(tr_sub.dataset)
        tr_ds.dataset.transform = ssl_tfm
        tr_dl = _make_loader(
            tr_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            seed=seed + 1000 * t,
            persistent_workers=False,  # avoid worker accumulation across many tasks
        )
        if verbose:
            print(f"  [{method}] Task {t}/{n_tasks-1}")
        if method == "pnr" and t > 0:
            model.reinit_projector()
        if method == "freeze_enc" and t > 0:
            for p in model.enc.parameters():
                p.requires_grad = False
        model.train()
        opt = _make_optimizer(model, optimizer, lr, method)
        _warmup = min(int(warmup_epochs), max(0, epochs - 1)) if warmup_epochs and warmup_epochs > 0 else 0
        remaining = epochs - _warmup if _warmup > 0 else epochs
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, remaining))
        if _warmup > 0:
            # Start from a small LR and ramp up linearly
            for g in opt.param_groups:
                g["lr"] = lr * (1.0 / _warmup)
        r_t = ppcl_reserve
        if method == "ppcl_adaptive":
            r_t = 0.2 * math.exp(-t / max(n_tasks - 1, 1))
        ppcl_pe_vals = []

        for ep in range(epochs):
            if _warmup > 0 and ep < _warmup:
                scale = float(ep + 1) / float(_warmup)
                for g in opt.param_groups:
                    g["lr"] = lr * scale

            for batch in tr_dl:
                (v1, v2), _ = batch
                v1, v2 = v1.to(device), v2.to(device)

                # LUMP mixup
                if method == "lump":
                    buf_imgs, _ = replay_buf.get_batch(v1.size(0))
                    if buf_imgs is not None:
                        v1, v2 = lump_mixup(v1, v2, buf_imgs, ssl_tfm)

                # Effective PPCL lambda with optional warmup/ramp
                _is_ppcl_family = method in (
                    "ppcl", "ppcl_adaptive", "ppcl_stable", "ppcl_mom",
                    "ppcl_rand", "ppcl_soft", "ppcl_enc",
                )
                _ppcl_skip = _is_ppcl_family and t < ppcl_warmup_tasks
                if _is_ppcl_family and not _ppcl_skip:
                    if t == ppcl_warmup_tasks and ppcl_lam_ramp_epochs > 0:
                        _eff_lam = ppcl_lam * min(1.0, (ep + 1) / ppcl_lam_ramp_epochs)
                    else:
                        _eff_lam = ppcl_lam
                else:
                    _eff_lam = ppcl_lam

                with autocast_ctx(use_amp):
                    if _ppcl_skip:
                        # Fall back to plain SimCLR during warmup tasks
                        _, z1, _ = model(v1)
                        _, z2, _ = model(v2)
                        loss = ntxent(z1, z2, TEMP)
                    elif method == "byol":
                        _, z1, p1 = model(v1)
                        _, z2, p2 = model(v2)
                        tz1 = model.forward_target(v1)
                        tz2 = model.forward_target(v2)
                        loss = byol_loss(p1, tz2, p2, tz1)
                    elif method == "byol_no_pred":
                        _, z1, _ = model(v1)
                        _, z2, _ = model(v2)
                        tz1 = model.forward_target(v1)
                        tz2 = model.forward_target(v2)
                        z1n = F.normalize(z1, dim=1)
                        z2n = F.normalize(z2, dim=1)
                        tz1n = F.normalize(tz1.detach(), dim=1)
                        tz2n = F.normalize(tz2.detach(), dim=1)
                        loss = 2 - (z1n * tz2n).sum(1).mean() - (z2n * tz1n).sum(1).mean()
                    elif method == "simsiam":
                        _, z1, p1 = model(v1)
                        _, z2, p2 = model(v2)
                        loss = simsiam_loss(p1, z2, p2, z1)
                    elif method == "simclr_pred":
                        _, z1, p1 = model(v1)
                        _, z2, p2 = model(v2)
                        loss = 0.5 * (ntxent(p1, z2.detach(), TEMP)
                                      + ntxent(p2, z1.detach(), TEMP))
                    elif method == "ppcl_mom":
                        _, z1, p1 = model(v1)
                        _, z2, p2 = model(v2)
                        tz1 = model.forward_target(v1)
                        tz2 = model.forward_target(v2)
                        loss_p, pe, _, _ = ppcl_loss(z1, z2, r_t, TEMP, _eff_lam)
                        ppcl_pe_vals.append(pe)
                        # BYOL auxiliary on the high-variance (active) dims only
                        D = z1.shape[1]
                        nr = max(1, int(D * r_t))
                        with torch.no_grad():
                            var = torch.cat([z1, z2], 0).var(dim=0)
                            _, lidx = var.topk(nr, largest=False)
                            hmask = torch.ones(D, dtype=torch.bool, device=device)
                            hmask[lidx] = False
                            hidx = torch.where(hmask)[0]
                        loss_b = byol_loss(p1[:, hidx], tz2[:, hidx], p2[:, hidx], tz1[:, hidx])
                        loss = loss_p + PPCL_MOM_BYOL_AUX_WEIGHT * loss_b
                    elif method in ("ppcl", "ppcl_adaptive"):
                        _, z1, _ = model(v1)
                        _, z2, _ = model(v2)
                        loss, pe, _, _ = ppcl_loss(z1, z2, r_t, TEMP, _eff_lam)
                        ppcl_pe_vals.append(pe)
                    elif method == "ppcl_soft":
                        _, z1, _ = model(v1)
                        _, z2, _ = model(v2)
                        loss, pe, _, _ = ppcl_loss_soft(z1, z2, TEMP, _eff_lam)
                        ppcl_pe_vals.append(pe)
                    elif method == "ppcl_enc":
                        h1, z1, _ = model(v1)
                        h2, z2, _ = model(v2)
                        # PPCL on 512-d encoder features (more dims available for the split)
                        loss_enc, pe, _, _ = ppcl_loss(h1, h2, r_t, TEMP, _eff_lam)
                        loss_proj = ntxent(z1, z2, TEMP)
                        loss = loss_proj + loss_enc
                        ppcl_pe_vals.append(pe)
                    elif method == "ppcl_rand":
                        _, z1, _ = model(v1)
                        _, z2, _ = model(v2)
                        if rand_low_idx is None:
                            raise RuntimeError("ppcl_rand requires rand_low_idx to be initialized.")
                        loss, pe, _, _ = ppcl_loss_fixed_low(z1, z2, rand_low_idx, TEMP, _eff_lam)
                        ppcl_pe_vals.append(pe)
                    elif method == "ppcl_stable":
                        _, z1, _ = model(v1)
                        _, z2, _ = model(v2)
                        if stable_sub is None:
                            raise RuntimeError("ppcl_stable requires VarianceEMASubspace state.")
                        loss, pe, _, _, _ = ppcl_loss_stable_subspace(
                            z1, z2, stable_sub, r_t, TEMP, _eff_lam, ntxent,
                            orth_lambda=ppcl_orth_lambda,
                        )
                        ppcl_pe_vals.append(pe)
                    elif method == "simclr_ema_l2":
                        _, z1, _ = model(v1)
                        _, z2, _ = model(v2)
                        if shadow_enc is None or shadow_proj is None:
                            raise RuntimeError("simclr_ema_l2 requires EMA shadow modules.")
                        loss = ntxent(z1, z2, TEMP)
                        sq = weight_modules_l2_squared_sum(model.enc, shadow_enc)
                        sq = sq + weight_modules_l2_squared_sum(model.proj, shadow_proj)
                        loss = loss + explicit_ema_l2_lambda * sq
                    elif method == "simclr_ema_l2_pred":
                        _, z1, p1 = model(v1)
                        _, z2, p2 = model(v2)
                        if shadow_enc is None or shadow_proj is None:
                            raise RuntimeError("simclr_ema_l2_pred requires EMA shadow modules.")
                        loss = 0.5 * (ntxent(p1, z2.detach(), TEMP)
                                      + ntxent(p2, z1.detach(), TEMP))
                        sq = weight_modules_l2_squared_sum(model.enc, shadow_enc)
                        sq = sq + weight_modules_l2_squared_sum(model.proj, shadow_proj)
                        loss = loss + explicit_ema_l2_lambda * sq
                    elif method == "barlow":
                        _, z1, _ = model(v1)
                        _, z2, _ = model(v2)
                        loss = barlow_loss(z1, z2)
                    elif method == "vicreg":
                        _, z1, _ = model(v1)
                        _, z2, _ = model(v2)
                        loss = vicreg_loss(z1, z2)
                    elif method == "ewc_ssl":
                        _, z1, _ = model(v1)
                        _, z2, _ = model(v2)
                        loss = ntxent(z1, z2, TEMP)
                        if fisher is not None:
                            ewc_pen = sum(
                                (fisher[n] * (p - prev_params[n]).pow(2)).sum()
                                for n, p in model.named_parameters() if n in fisher
                            )
                            loss += (EWC_LAM / 2) * ewc_pen
                    elif method == "lwf_ssl":
                        _, z1, _ = model(v1)
                        _, z2, _ = model(v2)
                        loss = ntxent(z1, z2, TEMP)
                        if prev_model is not None:
                            prev_model.eval()
                            with torch.no_grad():
                                _, pz1, _ = prev_model(v1)
                                _, pz2, _ = prev_model(v2)
                            loss += LWF_ALPHA * (distill_loss(z1, pz1) + distill_loss(z2, pz2))
                    elif method in ("replay_ssl", "cassle"):
                        _, z1, _ = model(v1)
                        _, z2, _ = model(v2)
                        loss = ntxent(z1, z2, TEMP)
                        buf_imgs, _ = replay_buf.get_batch(v1.size(0))
                        if buf_imgs is not None:
                            rb = buf_imgs.to(device)
                            _, rz, _ = model(rb)
                            loss += 0.5 * ntxent(rz, rz, TEMP)
                            if method == "cassle" and prev_model is not None:
                                prev_model.eval()
                                with torch.no_grad():
                                    _, prz, _ = prev_model(rb)
                                loss += 0.5 * (
                                    2 - 2 * (F.normalize(rz, dim=1)
                                             * F.normalize(prz.detach(), dim=1)).sum(1).mean()
                                )
                    elif method == "der_ssl":
                        _, z1, _ = model(v1)
                        _, z2, _ = model(v2)
                        loss = ntxent(z1, z2, TEMP)
                        buf_imgs, _, buf_reps = replay_buf.get_batch_with_reps(v1.size(0))
                        if buf_imgs is not None:
                            rb = buf_imgs.to(device)
                            _, rz, _ = model(rb)
                            loss += 0.5 * ntxent(rz, rz, TEMP)
                            if buf_reps is not None:
                                loss += 0.1 * F.mse_loss(rz, buf_reps.to(device))
                    elif method == "lump":
                        _, z1, _ = model(v1)
                        _, z2, _ = model(v2)
                        loss = ntxent(z1, z2, TEMP)
                    else:  # simclr, freeze_enc, pnr
                        _, z1, _ = model(v1)
                        _, z2, _ = model(v2)
                        loss = ntxent(z1, z2, TEMP)

                opt.zero_grad()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss: {float(loss.detach().cpu())}")
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if is_ema:
                    model.update_ema()
                elif (method in ("simclr_ema_l2", "simclr_ema_l2_pred")
                      and shadow_enc is not None and shadow_proj is not None):
                    ema_update_modules_(shadow_enc, model.enc, byol_mom)
                    ema_update_modules_(shadow_proj, model.proj, byol_mom)

            if _warmup > 0:
                if ep >= _warmup:
                    sched.step()
            else:
                sched.step()

        # Post-task bookkeeping
        if method == "ewc_ssl":
            fisher = {}
            model.eval()
            for n, p in model.named_parameters():
                if p.requires_grad:
                    fisher[n] = torch.zeros_like(p)
            cnt = 0
            for batch in tr_dl:
                if cnt > FISHER_MAX_BATCHES:
                    break
                (v1, v2), _ = batch
                v1, v2 = v1.to(device), v2.to(device)
                _, z1, _ = model(v1)
                _, z2, _ = model(v2)
                l = ntxent(z1, z2, TEMP)
                model.zero_grad()
                l.backward()
                for n, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.data.pow(2)
                cnt += v1.size(0)
            for n in fisher:
                fisher[n] /= max(cnt, 1)
            prev_params = {n: p.data.clone() for n, p in model.named_parameters()}
        if method in ("lwf_ssl", "cassle"):
            prev_model = copy.deepcopy(model)
        if method in ("replay_ssl", "lump", "cassle"):
            replay_buf.update(tr_sub, ev_tfm)
        if method == "der_ssl":
            replay_buf.update(tr_sub, ev_tfm, model, device)
        if method == "freeze_enc" and t > 0:
            for p in model.enc.parameters():
                p.requires_grad = True

        # CKA drift vs task-0 representation
        try:
            if t == 0:
                feat_snap, _ = _extract_batched(model, te_sub, ev_tfm, device)
                feat_snap = F.normalize(feat_snap, dim=1)
                cka_drifts.append(0.0)
            else:
                # On resume from task > 0, feat_snap won't be in memory; rebuild from task-0
                if feat_snap is None:
                    feat_snap, _ = _extract_batched(model, sp_te[0], ev_tfm, device)
                    feat_snap = F.normalize(feat_snap, dim=1)
                fc, _ = _extract_batched(model, sp_te[0], ev_tfm, device)
                fc = F.normalize(fc, dim=1)
                if feat_snap is None or fc is None:
                    cka_drifts.append(0.0)
                else:
                    n_s = min(feat_snap.size(0), fc.size(0))
                    cka_drifts.append(1.0 - compute_cka(feat_snap[:n_s], fc[:n_s]))
        except Exception as e:
            if verbose:
                print(f"  [cka] skipped at task {t}: {e}")
            cka_drifts.append(0.0)

        # Gradient alignment (replay methods)
        if method in ("replay_ssl", "lump", "der_ssl", "cassle") and t > 0:
            buf_imgs, _ = replay_buf.get_batch(batch_size)
            if buf_imgs is not None:
                model.train()
                rb = buf_imgs.to(device)
                _, rz, _ = model(rb)
                lr_loss = ntxent(rz, rz, TEMP)
                model.zero_grad()
                lr_loss.backward()
                g_r = torch.cat([
                    p.grad.flatten() for p in model.enc.parameters() if p.grad is not None
                ])
                _, tz1, _ = model(v1)
                _, tz2, _ = model(v2)
                tl = ntxent(tz1, tz2, TEMP)
                model.zero_grad()
                tl.backward()
                g_t = torch.cat([
                    p.grad.flatten() for p in model.enc.parameters() if p.grad is not None
                ])
                ga = F.cosine_similarity(g_r.unsqueeze(0), g_t.unsqueeze(0)).item()
                grad_aligns.append(ga)
            else:
                grad_aligns.append(0.0)
        else:
            grad_aligns.append(0.0)

        # Extract features once and reuse across all per-task metrics
        with torch.no_grad():
            _te_feats, _te_labs = _extract_batched(model, te_sub, ev_tfm, device)
            _te_feats_n = F.normalize(_te_feats, dim=1)
        sranks.append(compute_stable_rank(model))
        _s = torch.linalg.svdvals(_te_feats_n.float())
        _s_sum = _s.sum()
        if _s_sum < 1e-10:
            eranks.append(1.0)
        else:
            _p = _s / _s_sum
            _p = _p[_p > 1e-10]
            eranks.append(torch.exp(-(_p * _p.log()).sum()).item())
        uniforms.append(
            torch.pdist(_te_feats_n, p=2).pow(2).mul(-2).exp()
            .mean().clamp(min=1e-8).log().item()
        )
        _cls = _te_labs.unique()
        if len(_cls) >= 2:
            _cen = torch.stack([_te_feats_n[_te_labs == c].mean(0) for c in _cls])
            _inter = torch.pdist(_cen).mean().item()
            _intra = float(np.mean([
                torch.pdist(_te_feats_n[_te_labs == c]).mean().item()
                for c in _cls if (_te_labs == c).sum() > 1
            ]))
            lsis.append(_inter / (_intra + 1e-10))
        else:
            lsis.append(0.0)
        if method in ("ppcl", "ppcl_adaptive", "ppcl_mom", "ppcl_rand",
                       "ppcl_stable", "ppcl_soft", "ppcl_enc"):
            ppcl_eranks.append(float(np.mean(ppcl_pe_vals)) if ppcl_pe_vals else 0.0)
        else:
            ppcl_eranks.append(0.0)
        with torch.no_grad():
            if method in ("byol", "ppcl_mom", "byol_no_pred") and getattr(model, "use_ema", False):
                byol_target_l2_trace.append(float(byol_online_target_enc_proj_l2(model).item()))
            elif (method in ("simclr_ema_l2", "simclr_ema_l2_pred")
                  and shadow_enc is not None and shadow_proj is not None):
                byol_target_l2_trace.append(
                    float(simclr_shadow_enc_proj_l2(model, shadow_enc, shadow_proj).item())
                )
            else:
                byol_target_l2_trace.append(0.0)

        # kNN eval on all tasks seen so far
        accs = []
        for j in range(t + 1):
            acc = knn_eval(model, sp_tr[j], sp_te[j], ev_tfm, KNN_K, device)
            accs.append(acc)
        acc_matrix.append(accs)
        plast_ratios.append(accs[-1] / 100.0)
        avg = np.mean(accs)
        if verbose:
            print(f"    Task {t}: avg_acc={avg:.1f}% srank={sranks[-1]:.2f}")

        if autosave_every_task:
            partial = _empty_result(method, seed)
            partial.update({
                'avg_acc': np.mean(acc_matrix[-1]),
                'acc_matrix': acc_matrix.copy(),
                'task_seed': task_seed,
                'schema_version': RESULT_SCHEMA_VERSION,
                'torch_version': torch.__version__,
                'device': str(device),
                'forgetting': compute_forgetting(acc_matrix),
                'backward_transfer': compute_backward_transfer(acc_matrix),
                'fwd_transfer': compute_fwd_transfer_mean(fwd_transfer_scores),
                'fwd_transfer_per_task': list(fwd_transfer_scores),
                'sranks': sranks.copy(),
                'eranks': eranks.copy(),
                'uniforms': uniforms.copy(),
                'lsis': lsis.copy(),
                'ppcl_eranks': ppcl_eranks.copy(),
                'cka_drifts': cka_drifts.copy(),
                'grad_aligns': grad_aligns.copy(),
                'plast_ratios': plast_ratios.copy(),
                'byol_target_weight_l2': list(byol_target_l2_trace),
                'explicit_ema_l2_lambda': explicit_ema_l2_lambda if method == "simclr_ema_l2" else 0.0,
                'time_min': (time.time() - t0) / 60,
                'n_params': n_params,
                'extra_params': extra_params,
            })
            save_result(partial, f"{(ckpt_tag or 'run')}_partial_t{t}")
        if save_ckpt and save_ckpt_each_task and ckpt_tag:
            save_checkpoint(model, method, seed, f"{ckpt_tag}_task{t}")

    # Linear probe (optional secondary eval for camera-ready tables)
    linear_acc = 0.0
    linear_accs = []
    if eval_linear_probe:
        try:
            linear_acc, linear_accs = linear_probe_eval(
                model, sp_tr, sp_te, n_tasks, ev_tfm, nc, device
            )
        except Exception as e:
            print(f"  [linear_probe] failed: {e}")

    peak_memory_mb = 0.0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(device)

    result = _empty_result(method, seed)
    import platform
    import torchvision
    result.update({
        'avg_acc': np.mean(acc_matrix[-1]),
        'acc_matrix': acc_matrix,
        'task_seed': task_seed,
        'schema_version': RESULT_SCHEMA_VERSION,
        'torch_version': torch.__version__,
        'torchvision_version': torchvision.__version__,
        'python_version': platform.python_version(),
        'device': str(device),
        'forgetting': compute_forgetting(acc_matrix),
        'backward_transfer': compute_backward_transfer(acc_matrix),
        'fwd_transfer': compute_fwd_transfer_mean(fwd_transfer_scores),
        'fwd_transfer_per_task': list(fwd_transfer_scores),
        'sranks': sranks,
        'eranks': eranks,
        'uniforms': uniforms,
        'lsis': lsis,
        'ppcl_eranks': ppcl_eranks,
        'cka_drifts': cka_drifts,
        'grad_aligns': grad_aligns,
        'plast_ratios': plast_ratios,
        'byol_target_weight_l2': list(byol_target_l2_trace),
        'explicit_ema_l2_lambda': (
            explicit_ema_l2_lambda
            if method in ("simclr_ema_l2", "simclr_ema_l2_pred") else 0.0
        ),
        'linear_acc': linear_acc,
        'linear_accs': linear_accs,
        'peak_memory_mb': peak_memory_mb,
        'time_min': (time.time() - t0) / 60,
        'n_params': n_params,
        'extra_params': extra_params,
    })
    if save_ckpt and ckpt_tag:
        save_checkpoint(model, method, seed, ckpt_tag)
    return result, model


def _run_oracle(model, sp_tr, sp_te, n_tasks, ssl_tfm, ev_tfm, nc,
                epochs, seed, device, use_amp, scaler, n_params, t0, save_ckpt, ckpt_tag):
    # Combine all task subsets without pre-applied transforms
    raw_subs = []
    for t in range(n_tasks):
        raw_sub = copy.copy(sp_tr[t])
        raw_sub.dataset = copy.copy(sp_tr[t].dataset)
        raw_sub.dataset.transform = None
        raw_subs.append(raw_sub)
    combined = ConcatDataset(raw_subs)

    class _W(torch.utils.data.Dataset):
        def __init__(self, ds, tfm):
            self.ds = ds
            self.tfm = tfm

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            img, lab = self.ds[i]
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            return self.tfm(img), lab

    wrapped = _W(combined, ssl_tfm)
    dl = _make_loader(wrapped, batch_size=256, shuffle=True, num_workers=2,
                      pin_memory=True, drop_last=True, seed=seed,
                      persistent_workers=False)
    model.train()
    opt = _make_optimizer(model, "auto", LR, "simclr")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print("  [oracle] Training on all tasks jointly...")
    for ep in range(epochs):
        for (v1, v2), _ in dl:
            v1, v2 = v1.to(device), v2.to(device)
            with autocast_ctx(use_amp):
                _, z1, _ = model(v1)
                _, z2, _ = model(v2)
                loss = ntxent(z1, z2, TEMP)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
    avg_acc, per_task = linear_probe_eval(model, sp_tr, sp_te, n_tasks, ev_tfm, nc, device)
    knn_accs = []
    for j in range(n_tasks):
        knn_accs.append(knn_eval(model, sp_tr[j], sp_te[j], ev_tfm, KNN_K, device))
    knn_avg = np.mean(knn_accs)
    print(f"  [oracle] linear_probe={avg_acc:.1f}% knn={knn_avg:.1f}%")
    result = _empty_result("oracle", seed)
    result.update({
        'avg_acc': avg_acc,
        'forgetting': 0.0,
        'backward_transfer': 0.0,
        'fwd_transfer': 0.0,
        'fwd_transfer_per_task': [],
        'byol_target_weight_l2': [],
        'explicit_ema_l2_lambda': 0.0,
        'schema_version': RESULT_SCHEMA_VERSION,
        'task_seed': seed,
        'torch_version': torch.__version__,
        'device': str(device),
        'acc_matrix': [per_task],
        'sranks': [compute_stable_rank(model)],
        'eranks': [compute_erank(model, sp_te[0], ev_tfm, device)],
        'uniforms': [compute_uniformity(model, sp_te[0], ev_tfm, device)],
        'lsis': [compute_lsi(model, sp_te[0], ev_tfm, device)],
        'linear_acc': avg_acc,
        'linear_accs': per_task,
        'time_min': (time.time() - t0) / 60,
        'n_params': n_params,
        'extra_params': 0,
    })
    if save_ckpt and ckpt_tag:
        save_checkpoint(model, "oracle", seed, ckpt_tag)
    return result, model


def _run_supervised(model, sp_tr, sp_te, n_tasks, ev_tfm, nc,
                    epochs, seed, device, use_amp, scaler, n_params, t0):
    clf = nn.Linear(ENC_DIM, nc).to(device)
    acc_matrix = []
    for t in range(n_tasks):
        tr_sub = sp_tr[t]
        tr_ds = copy.copy(tr_sub)
        tr_ds.dataset = copy.copy(tr_sub.dataset)
        tr_ds.dataset.transform = ev_tfm
        dl = _make_loader(tr_ds, batch_size=256, shuffle=True, num_workers=2,
                          pin_memory=True, drop_last=True, seed=seed + 1000 * t,
                          persistent_workers=False)
        model.train()
        clf.train()
        all_p = list(model.enc.parameters()) + list(clf.parameters())
        opt = torch.optim.SGD(all_p, lr=LR, momentum=0.9, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for ep in range(epochs):
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                with autocast_ctx(use_amp):
                    h = model.enc(xb)
                    loss = F.cross_entropy(clf(h), yb)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(all_p, 1.0)
                scaler.step(opt)
                scaler.update()
            sched.step()
        model.eval()
        clf.eval()
        accs = []
        for j in range(t + 1):
            te_ds = copy.copy(sp_te[j])
            te_ds.dataset = copy.copy(sp_te[j].dataset)
            te_ds.dataset.transform = ev_tfm
            te_dl = _make_loader(te_ds, batch_size=256, shuffle=False,
                                  num_workers=2, pin_memory=True)
            correct = total = 0
            with torch.no_grad():
                for xb, yb in te_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = clf(model.enc(xb)).argmax(1)
                    correct += (pred == yb).sum().item()
                    total += yb.size(0)
            accs.append(correct / total * 100)
        acc_matrix.append(accs)
    result = _empty_result("supervised", seed)
    result.update({
        'avg_acc': np.mean(acc_matrix[-1]),
        'acc_matrix': acc_matrix,
        'schema_version': RESULT_SCHEMA_VERSION,
        'task_seed': seed,
        'torch_version': torch.__version__,
        'device': str(device),
        'forgetting': compute_forgetting(acc_matrix),
        'backward_transfer': compute_backward_transfer(acc_matrix),
        'fwd_transfer': 0.0,
        'fwd_transfer_per_task': [],
        'byol_target_weight_l2': [0.0] * n_tasks,
        'explicit_ema_l2_lambda': 0.0,
        'sranks': [compute_stable_rank(model)] * n_tasks,
        'linear_acc': np.mean(acc_matrix[-1]),
        'linear_accs': acc_matrix[-1],
        'time_min': (time.time() - t0) / 60,
        'n_params': n_params,
        'extra_params': 0,
    })
    return result, model


def _eval_only(model, method, sp_tr, sp_te, n_tasks, ev_tfm, nc, seed, device=None):
    """kNN eval pass on an existing model (no training). BWT is always 0.0."""
    device = device or DEVICE
    model.eval()
    acc_matrix = []
    for t in range(n_tasks):
        accs = []
        for j in range(t + 1):
            acc = knn_eval(model, sp_tr[j], sp_te[j], ev_tfm, KNN_K, device)
            accs.append(acc)
        acc_matrix.append(accs)
    result = _empty_result(method, seed)
    result.update({
        'avg_acc': np.mean(acc_matrix[-1]),
        'acc_matrix': acc_matrix,
        'schema_version': RESULT_SCHEMA_VERSION,
        'task_seed': seed,
        'torch_version': torch.__version__,
        'device': str(device),
        'forgetting': compute_forgetting(acc_matrix),
        'backward_transfer': compute_backward_transfer(acc_matrix),
        'fwd_transfer': 0.0,
        'fwd_transfer_per_task': [],
        'byol_target_weight_l2': [0.0] * n_tasks,
        'explicit_ema_l2_lambda': 0.0,
        'sranks': [compute_stable_rank(model)] * n_tasks,
        'eranks': [compute_erank(model, sp_te[t], ev_tfm, device) for t in range(n_tasks)],
        'uniforms': [compute_uniformity(model, sp_te[t], ev_tfm, device) for t in range(n_tasks)],
        'lsis': [compute_lsi(model, sp_te[t], ev_tfm, device) for t in range(n_tasks)],
        'linear_acc': np.mean(acc_matrix[-1]),
        'linear_accs': acc_matrix[-1],
        'n_params': sum(p.numel() for p in model.parameters()),
    })
    return result, model


# --- Summary ---
def print_summary(results_dict):
    print(f"\n{'Method':<16} {'AvgAcc':>8} {'Forget':>8} {'BWT':>8} {'FwdTf':>8} {'SRank':>7} {'eRank':>7} {'LSI':>7}")
    print("-" * 74)
    for tag, r in sorted(results_dict.items()):
        sr = np.mean(r.get('sranks', [])) if r.get('sranks') else 0
        er = np.mean(r.get('eranks', [])) if r.get('eranks') else 0
        ls = np.mean(r.get('lsis', [])) if r.get('lsis') else 0
        bwt = r.get('backward_transfer', 0.0)
        print(
            f"{r.get('method', '?'):<16} {r.get('avg_acc', 0):>7.1f}% "
            f"{r.get('forgetting', 0):>7.1f}% {bwt:>7.2f}% "
            f"{r.get('fwd_transfer', 0):>7.1f}% "
            f"{sr:>6.2f} {er:>6.2f} {ls:>6.2f}"
        )


def compute_efficiency_table(phase2_results):
    """Print compute efficiency relative to SimCLR."""
    base = None
    for tag, r in phase2_results.items():
        if r.get('method') == 'simclr' and r.get('seed') == 0:
            base = r
            break
    if base is None:
        print("No SimCLR baseline found.")
        return
    bt = base.get('time_min', 1)
    print(f"\n{'Method':<16} {'Time(min)':>10} {'Rel.Time':>10} {'ExtraParams':>12} {'MemOverhead':>12}")
    print("-" * 62)
    seen = set()
    for tag, r in sorted(phase2_results.items()):
        m = r.get('method', '?')
        if m in seen:
            continue
        seen.add(m)
        t = r.get('time_min', 0)
        ep = r.get('extra_params', 0)
        print(
            f"{m:<16} {t:>9.1f} {t/bt:>9.2f}x {ep:>11,} "
            f"{'~2x' if ep > 0 else '1x':>12}"
        )


print("[core] All modules loaded.")
