import os
import sys
import time
import argparse
from contextlib import nullcontext
from collections import defaultdict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
import tiktoken

# sys.path.insert(0, '/Users/alimirz/Research/EEG_FM/EEG_FM')
sys.path.insert(0, '/home/alimirz/2026/EEG_FM/EEG_FM/')
from data_split_scripts.probe_label_hunter import ProbeLabelHunterV3

from model.model_neurolm import NeuroLM
from model.model import GPTConfig
from dataset import standard_1020
from utils import cosine_scheduler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EEG_DATA_DIR = '/orcd/compute/dinaktbi/001/2026/EEG_FM/preprocessed_eeg_v2'

# 19-channel order used by preprocessed_eeg_v2 (lowercase, as stored in H5)
CHANNEL_ORDER = ['o1', 'o2', 't6', 'p4', 'pz', 'p3', 't5', 't3',
                 'c3', 'cz', 'c4', 't4', 'f8', 'f4', 'fz', 'f3', 'f7', 'fp1', 'fp2']
CHANNEL_ORDER_UPPER = [ch.upper() for ch in CHANNEL_ORDER]

FS = 200
WINDOW_SECONDS = 30
WINDOW_SAMPLES = WINDOW_SECONDS * FS   # 6000
NUM_CHANS = len(CHANNEL_ORDER)         # 19
NUM_TIME = WINDOW_SECONDS              # 30
EEG_MAX_LEN = NUM_TIME * NUM_CHANS     # 570
TEXT_MAX_LEN = 80
PATCH_SIZE = 200                       # samples per EEG patch (1 second at 200 Hz)

# Precomputed channel / time index arrays (shared across dataset instances)
_CHAN_INDICES = [standard_1020.index(ch) for ch in CHANNEL_ORDER_UPPER] * NUM_TIME
_TIME_INDICES = [t for t in range(NUM_TIME) for _ in range(NUM_CHANS)]

master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None


# ---------------------------------------------------------------------------
# Distributed / dtype init
# ---------------------------------------------------------------------------

def init(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank
    device = 'cuda'
    dtype = ('bfloat16'
             if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             else 'float16')

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32,
               'bfloat16': torch.bfloat16,
               'float16': torch.float16}[dtype]
    ctx = (nullcontext() if device_type == 'cpu'
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))


# ---------------------------------------------------------------------------
# Task → question string
# ---------------------------------------------------------------------------

def task_to_question(task_name: str) -> str:
    prefix, name = task_name.split('_', 1)
    if prefix in ('med', 'smed'):
        return f"Is the patient taking {name}?"
    elif prefix in ('dis', 'cond', 'diag'):
        return f"Does the patient have {name}?"
    elif prefix == 'feat':
        return f"Does the EEG show {name}?"
    else:
        return f"Does this EEG relate to {name}?"


def build_prompt(task_name: str) -> str:
    return f"[SEP] Question: {task_to_question(task_name)} Answer:"


def build_full_text(task_name: str, label: int) -> str:
    answer = " Yes" if label == 1 else " No"
    return build_prompt(task_name) + answer + " <|endoftext|>"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InternalInstructDataset(Dataset):
    """
    Train mode: one item per recording file.
      - Random 30-second segment chosen fresh each __getitem__ call.
      - Random task with non-(-1) label chosen per call.
      - Returns (X_eeg, X_text, Y_text, input_chans, input_time, eeg_mask, gpt_mask).

    Val mode: one item per (recording, task) pair where label != -1.
      - Fixed 30-second segment per recording (deterministic seed).
      - Returns (X_eeg, X_text, label, input_chans, input_time, eeg_mask, gpt_mask)
        where X_text is the prompt only (no answer), for logit-based evaluation.
    """

    def __init__(self, filenames, all_labels, downstream_tasks,
                 eeg_data_dir=EEG_DATA_DIR, mode='train', seed=42):
        assert mode in ('train', 'val')
        self.filenames = list(filenames)
        self.all_labels = all_labels          # (N_files, N_tasks) float tensor
        self.tasks = list(downstream_tasks)
        self.eeg_data_dir = eeg_data_dir
        self.mode = mode
        self.seed = seed

        enc = tiktoken.get_encoding("gpt2")
        self.yes_id = enc.encode(" Yes")[0]
        self.no_id = enc.encode(" No")[0]

        # Precompute prompt token tensors (fixed per task)
        self.prompt_tokens: dict[str, torch.Tensor] = {}
        for task in self.tasks:
            ids = enc.encode(build_prompt(task))
            self.prompt_tokens[task] = torch.tensor(ids, dtype=torch.long)

        # Precompute full-text token tensors for both labels (train only)
        self._full_text_tokens: dict[tuple, torch.Tensor] = {}
        for task in self.tasks:
            for lbl in (0, 1):
                ids = enc.encode(build_full_text(task, lbl), allowed_special={'<|endoftext|>'})
                ids_t = torch.tensor(ids[:TEXT_MAX_LEN], dtype=torch.long)
                self._full_text_tokens[(task, lbl)] = ids_t

        if mode == 'val':
            # One deterministic valid task per file (seeded per file index)
            self.val_task_assignments: list[tuple[int, int]] = []
            for fi in range(len(self.filenames)):
                file_labels = self.all_labels[fi]
                valid = torch.where(file_labels != -1)[0]
                rng = np.random.default_rng(self.seed + fi + 10000)
                if len(valid) > 0:
                    task_idx = int(valid[int(rng.integers(0, len(valid)))].item())
                    label = int(file_labels[task_idx].item())
                else:
                    task_idx, label = 0, 0
                self.val_task_assignments.append((task_idx, label))

            # Fixed 30s offsets per file — scan H5 headers once at init
            self.val_offsets = self._precompute_val_offsets()

    def _precompute_val_offsets(self) -> dict[int, int]:
        offsets: dict[int, int] = {}
        for fi, fname in enumerate(self.filenames):
            path = os.path.join(self.eeg_data_dir, fname)
            try:
                with h5py.File(path, 'r') as f:
                    file_len = f['recording/data'].shape[0]
            except Exception:
                file_len = WINDOW_SAMPLES
            rng = np.random.default_rng(self.seed + fi)
            max_off = max(0, file_len - WINDOW_SAMPLES)
            offsets[fi] = int(rng.integers(0, max_off + 1)) if max_off > 0 else 0
        return offsets

    def _load_eeg_window(self, file_idx: int, offset: int) -> torch.Tensor:
        path = os.path.join(self.eeg_data_dir, self.filenames[file_idx])
        with h5py.File(path, 'r') as f:
            raw = f['recording/data'][offset:offset + WINDOW_SAMPLES, :]
            ch_names_raw = [
                (c.decode() if isinstance(c, bytes) else c).lower()
                for c in f['recording/ch_names'][:]
            ]

        ch_map = {ch: i for i, ch in enumerate(ch_names_raw)}
        ordered = np.zeros((WINDOW_SAMPLES, NUM_CHANS), dtype=np.float32)
        for c_out, ch in enumerate(CHANNEL_ORDER):
            if ch in ch_map:
                ordered[:, c_out] = raw[:, ch_map[ch]]

        # Per-channel z-score + clip (consistent with downstream pipeline)
        mean = ordered.mean(axis=0, keepdims=True)
        std = ordered.std(axis=0, keepdims=True) + 1e-8
        ordered = np.clip((ordered - mean) / std, -15.0, 15.0)

        # (WINDOW_SAMPLES, 19) → (NUM_TIME, PATCH_SIZE, 19) → (NUM_TIME, 19, PATCH_SIZE)
        # → (EEG_MAX_LEN, PATCH_SIZE)
        patches = ordered.reshape(NUM_TIME, PATCH_SIZE, NUM_CHANS)
        patches = patches.transpose(0, 2, 1).reshape(-1, PATCH_SIZE)   # (570, 200)
        return torch.tensor(patches, dtype=torch.float32)

    def _build_train_text(self, task_name: str, label: int):
        """Returns (X_text, Y_text) both padded to TEXT_MAX_LEN."""
        full_ids = self._full_text_tokens[(task_name, label)]   # already clipped to TEXT_MAX_LEN
        prompt_len = self.prompt_tokens[task_name].size(0)
        valid_len = full_ids.size(0)

        # Pad with EOT (50256)
        X_text = torch.full((TEXT_MAX_LEN,), fill_value=50256, dtype=torch.long)
        X_text[:valid_len] = full_ids

        # Y_text: -1 everywhere; answer tokens placed at next-token offset
        Y_text = torch.full((TEXT_MAX_LEN,), fill_value=-1, dtype=torch.long)
        ans_start = prompt_len - 1      # position whose target is the first answer token
        ans_end = valid_len - 1
        if ans_start < TEXT_MAX_LEN and ans_end > ans_start:
            Y_text[ans_start:ans_end] = X_text[ans_start + 1:ans_end + 1]

        return X_text, Y_text

    @staticmethod
    def _build_gpt_mask(n_text: int) -> torch.Tensor:
        """Stair-stepping causal mask for (EEG_MAX_LEN + n_text) positions."""
        n_total = EEG_MAX_LEN + n_text
        mask = torch.tril(torch.ones(n_total, n_total)).unsqueeze(0)   # (1, N, N)
        for t in range(NUM_TIME):
            s, e = t * NUM_CHANS, (t + 1) * NUM_CHANS
            mask[0, s:e, s:e] = 1.0
        return mask.bool()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx):
        input_chans = torch.IntTensor(_CHAN_INDICES)
        input_time = torch.IntTensor(_TIME_INDICES)
        eeg_mask = torch.ones(EEG_MAX_LEN, dtype=torch.bool)

        if self.mode == 'val':
            file_idx = idx
            task_idx, label = self.val_task_assignments[file_idx]
            offset = self.val_offsets[file_idx]
            task_name = self.tasks[task_idx]

            X_eeg = self._load_eeg_window(file_idx, offset)
            X_text = self.prompt_tokens[task_name].clone()    # exact prompt, no padding
            gpt_mask = self._build_gpt_mask(X_text.size(0))
            return (X_eeg, X_text, torch.tensor(label, dtype=torch.long),
                    input_chans, input_time, eeg_mask, gpt_mask)

        else:
            file_idx = idx
            path = os.path.join(self.eeg_data_dir, self.filenames[file_idx])
            try:
                with h5py.File(path, 'r') as f:
                    file_len = f['recording/data'].shape[0]
            except Exception:
                file_len = WINDOW_SAMPLES
            max_off = max(0, file_len - WINDOW_SAMPLES)
            offset = int(np.random.randint(0, max_off + 1)) if max_off > 0 else 0

            # Random valid task for this file
            file_labels = self.all_labels[file_idx]
            valid = torch.where(file_labels != -1)[0]
            if len(valid) == 0:
                task_idx = int(np.random.randint(0, len(self.tasks)))
                label = 0
            else:
                task_idx = int(valid[np.random.randint(0, len(valid))].item())
                label = int(file_labels[task_idx].item())
            task_name = self.tasks[task_idx]

            X_eeg = self._load_eeg_window(file_idx, offset)
            X_text, Y_text = self._build_train_text(task_name, label)
            gpt_mask = self._build_gpt_mask(TEXT_MAX_LEN)
            return X_eeg, X_text, Y_text, input_chans, input_time, eeg_mask, gpt_mask


# ---------------------------------------------------------------------------
# Safe loading wrappers
# ---------------------------------------------------------------------------

class SafeDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            return self.dataset[idx]
        except Exception as e:
            print(f"[SafeDataset] skipping idx {idx}: {e}")
            return None


def safe_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
        print(f"[safe_collate_fn] collate error: {e}")
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_dataset, writer, global_step: int) -> dict:
    """
    For each binary task: collects (yes_logit - no_logit) scores and ground-truth
    labels, then computes balanced_accuracy, ROC-AUC, and PR-AUC.

    Items within the same task always share an identical prompt → same X_text
    length, so we batch them together for efficiency.
    """
    model.eval()

    # Unwrap SafeDataset if present
    inner: InternalInstructDataset = (
        val_dataset.dataset if isinstance(val_dataset, SafeDataset) else val_dataset
    )

    yes_id = inner.yes_id
    no_id = inner.no_id

    # Group val files by their assigned task index for efficient batched inference
    task_groups: dict[int, list[int]] = defaultdict(list)
    for fi, (task_idx, _) in enumerate(inner.val_task_assignments):
        task_groups[task_idx].append(fi)

    task_scores: dict[str, list[float]] = defaultdict(list)
    task_labels_map: dict[str, list[int]] = defaultdict(list)

    for task_idx, file_indices in task_groups.items():
        task_name = inner.tasks[task_idx]
        # All files in this group share the same X_text (fixed prompt per task)
        prompt_tokens = inner.prompt_tokens[task_name]
        n_text = prompt_tokens.size(0)
        gpt_mask = InternalInstructDataset._build_gpt_mask(n_text).to(device)

        # Process in mini-batches
        for batch_start in range(0, len(file_indices), 32):
            batch_fis = file_indices[batch_start:batch_start + 32]
            eegs, labels_batch = [], []
            for fi in batch_fis:
                _, gt_label = inner.val_task_assignments[fi]
                offset = inner.val_offsets[fi]
                try:
                    eegs.append(inner._load_eeg_window(fi, offset))
                    labels_batch.append(gt_label)
                except Exception as e:
                    print(f"[evaluate] skipping val file {fi}: {e}")
            if not eegs:
                continue

            B = len(eegs)
            X_eeg = torch.stack(eegs).float().to(device)                          # (B, 570, 200)
            X_text = prompt_tokens.unsqueeze(0).expand(B, -1).to(device)          # (B, prompt_len)
            input_chans = torch.IntTensor(_CHAN_INDICES).unsqueeze(0).expand(B, -1).to(device)
            input_time = torch.IntTensor(_TIME_INDICES).unsqueeze(0).expand(B, -1).to(device)
            eeg_mask = torch.ones(B, EEG_MAX_LEN, dtype=torch.bool, device=device)
            batch_gpt_mask = gpt_mask.expand(B, -1, -1, -1)                       # (B, 1, N, N)

            Y_eeg = torch.full((B, EEG_MAX_LEN),
                               fill_value=-1 - model.GPT2.config.vocab_size,
                               device=device)

            with ctx:
                _, _, logits = model(X_eeg, Y_eeg, X_text, None,
                                     input_chans, input_time, eeg_mask,
                                     eeg_text_mask=batch_gpt_mask)
            # logits: (B, 1, vocab_size)  — last-position only
            last = logits[:, 0, :]                                # (B, vocab_size)
            scores = (last[:, yes_id] - last[:, no_id]).cpu().tolist()
            task_scores[task_name].extend(scores)
            task_labels_map[task_name].extend(labels_batch)

    results = {}
    for task_name in val_dataset.tasks:
        scores = np.array(task_scores.get(task_name, []))
        labels = np.array(task_labels_map.get(task_name, []))
        if len(scores) == 0 or len(np.unique(labels)) < 2:
            continue
        binary_preds = (scores > 0).astype(int)
        try:
            bal_acc = balanced_accuracy_score(labels, binary_preds)
            roc = roc_auc_score(labels, scores)
            pr = average_precision_score(labels, scores)
        except Exception:
            bal_acc, roc, pr = 0.0, 0.5, 0.5
        results[task_name] = {'balanced_acc': bal_acc, 'roc_auc': roc, 'pr_auc': pr}
        if writer is not None:
            writer.add_scalar(f'val/{task_name}/balanced_acc', bal_acc, global_step)
            writer.add_scalar(f'val/{task_name}/roc_auc', roc, global_step)
            writer.add_scalar(f'val/{task_name}/pr_auc', pr, global_step)

    if results:
        mean_bal = np.mean([v['balanced_acc'] for v in results.values()])
        mean_roc = np.mean([v['roc_auc'] for v in results.values()])
        mean_pr = np.mean([v['pr_auc'] for v in results.values()])
        results['__mean__'] = {'balanced_acc': mean_bal, 'roc_auc': mean_roc, 'pr_auc': mean_pr}
        if writer is not None:
            writer.add_scalar('val/mean_balanced_acc', mean_bal, global_step)
            writer.add_scalar('val/mean_roc_auc', mean_roc, global_step)
            writer.add_scalar('val/mean_pr_auc', mean_pr, global_step)

    model.train()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints/instruction-internal')
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    writer = None
    if master_process:
        writer = SummaryWriter(os.path.join(args.out_dir, 'runs/instruction-internal'))

    # OpenWebText text-LM batches
    text_dir = os.path.join(args.out_dir, 'text')

    def get_batch(split):
        data = np.memmap(
            os.path.join(text_dir, 'train.bin' if split == 'train' else 'val.bin'),
            dtype=np.uint16, mode='r',
        )
        ix = torch.randint(len(data) - args.block_size, (args.text_batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + args.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + args.block_size].astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # Labels and file splits
    if master_process:
        print("Loading ProbeLabelHunterV3 ...")
    labels = ProbeLabelHunterV3(EEG_DATA_DIR, debug=args.debug,
                                overwrite=False, paper_version=True)
    downstream_tasks = labels.downstream_tasks
    if master_process:
        print(f"{len(downstream_tasks)} tasks: {downstream_tasks[:5]} ...")

    train_labels = labels.get_labels(labels.train_filenames)
    val_labels = labels.get_labels(labels.val_filenames)

    train_dataset = SafeDataset(InternalInstructDataset(
        labels.train_filenames, train_labels, downstream_tasks,
        eeg_data_dir=EEG_DATA_DIR, mode='train', seed=args.seed,
    ))
    val_dataset = SafeDataset(InternalInstructDataset(
        labels.val_filenames, val_labels, downstream_tasks,
        eeg_data_dir=EEG_DATA_DIR, mode='val', seed=args.seed,
    ))
    if master_process:
        print(f"Train files: {len(train_dataset)}  |  Val items: {len(val_dataset)}")

    if ddp:
        sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True,
        )
        train_loader = DataLoader(train_dataset, sampler=sampler,
                                  batch_size=args.eeg_batch_size,
                                  num_workers=8, pin_memory=True, drop_last=True,
                                  collate_fn=safe_collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.eeg_batch_size,
                                  shuffle=True, num_workers=8,
                                  pin_memory=True, drop_last=True,
                                  collate_fn=safe_collate_fn)

    # Model init
    iter_num = 0
    n_layer, n_head, n_embd = 12, 12, 768
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                      block_size=args.block_size, bias=False,
                      vocab_size=50257, dropout=0.0)

    resume_path = os.path.join(checkpoint_out_dir, 'ckpt.pt')
    if os.path.exists(resume_path):
        if master_process:
            print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint['model_args'][k]
        gptconf = GPTConfig(**model_args)
        model = NeuroLM(gptconf, init_from='scratch')
        _load_state_dict(model, checkpoint['model'])
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1
    else:
        if master_process:
            print(f"Loading pretrained NeuroLM from {args.neurolm_path}")
        checkpoint = torch.load(args.neurolm_path, map_location=device, weights_only=False)
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint['model_args'][k]
        gptconf = GPTConfig(**model_args)
        model = NeuroLM(gptconf, init_from='scratch')
        _load_state_dict(model, checkpoint['model'])
        start_epoch = 0

    model.to(device)
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type,
    )
    if os.path.exists(resume_path):
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None  # free memory

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    num_steps = len(train_dataset) // args.eeg_batch_size // ddp_world_size
    lr_schedule = cosine_scheduler(
        args.learning_rate, args.min_lr, args.epochs, num_steps,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=int(args.warmup_ratio * num_steps * args.epochs),
    )

    X_text2, Y_text2 = get_batch('train')
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        if ddp:
            train_loader.sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue

            lr = lr_schedule[iter_num] if args.decay_lr else args.learning_rate
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            X_eeg, X_text, Y_text, input_chans, input_time, eeg_mask, gpt_mask = batch
            X_eeg = X_eeg.float().to(device, non_blocking=True)
            X_text = X_text.to(device, non_blocking=True)
            Y_text = Y_text.to(device, non_blocking=True)
            input_chans = input_chans.to(device, non_blocking=True)
            input_time = input_time.to(device, non_blocking=True)
            eeg_mask = eeg_mask.to(device, non_blocking=True)
            gpt_mask = gpt_mask.to(device, non_blocking=True)

            Y_eeg = torch.full((X_eeg.size(0), X_eeg.size(1)),
                               fill_value=-1 - raw_model.GPT2.config.vocab_size,
                               device=device)

            if ddp:
                model.require_backward_grad_sync = (
                    (step + 1) % args.gradient_accumulation_steps == 0
                )

            with ctx:
                loss1, log1, _ = model(X_eeg, Y_eeg, X_text, Y_text,
                                       input_chans, input_time, eeg_mask,
                                       eeg_text_mask=gpt_mask)
                loss2, log2, _ = model(None, None, X_text2, Y_text2)
                loss = (loss1 + loss2) / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            X_text2, Y_text2 = get_batch('train')

            if (iter_num + 1) % args.log_interval == 0 and master_process:
                t1 = time.time()
                instr_loss = log1['train/loss']
                text_loss = log2['train/loss']
                instr_acc = log1.get('train/accuracy', float('nan'))
                print(f"epoch {epoch}  step [{step+1}/{num_steps}]  "
                      f"total={instr_loss+text_loss:.4f}  "
                      f"instr={instr_loss:.4f}  text={text_loss:.4f}  "
                      f"acc={instr_acc:.3f}  lr={lr:.2e}  "
                      f"dt={(t1-t0)*1000:.0f}ms")
                writer.add_scalar('train/total_loss', instr_loss + text_loss, iter_num)
                writer.add_scalar('train/instruction_loss', instr_loss, iter_num)
                writer.add_scalar('train/text_loss', text_loss, iter_num)
                writer.add_scalar('train/instruction_accuracy', instr_acc, iter_num)
                writer.add_scalar('train/lr', lr, iter_num)
                t0 = t1

            iter_num += 1

        # Checkpoint (only on final epoch if --save_checkpoint is set)
        if master_process and args.save_checkpoint and epoch == args.epochs - 1:
            ckpt = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'epoch': epoch,
            }
            torch.save(ckpt, os.path.join(checkpoint_out_dir, 'ckpt.pt'))
            print(f"Checkpoint saved to {checkpoint_out_dir}/ckpt.pt")

        # Validation (master only — avoids DDP sync complexity)
        if master_process:
            results = evaluate(raw_model, val_dataset, writer, global_step=iter_num)
            _print_val_results(epoch, results)

    if writer is not None:
        writer.close()
    if ddp:
        destroy_process_group()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_state_dict(model, state_dict):
    prefix = '_orig_mod.'
    cleaned = {(k[len(prefix):] if k.startswith(prefix) else k): v
               for k, v in state_dict.items()}
    model.load_state_dict(cleaned)


def _print_val_results(epoch: int, results: dict):
    print(f"\n=== Val results (epoch {epoch}) ===")
    if '__mean__' in results:
        m = results['__mean__']
        print(f"  [mean]  bal_acc={m['balanced_acc']:.4f}  "
              f"roc={m['roc_auc']:.4f}  pr={m['pr_auc']:.4f}")
    for name, m in results.items():
        if name == '__mean__':
            continue
        print(f"  {name}: bal_acc={m['balanced_acc']:.4f}  "
              f"roc={m['roc_auc']:.4f}  pr={m['pr_auc']:.4f}")
    print()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser('NeuroLM internal instruction tuning')
    p.add_argument('--out_dir', default='./',
                   help='Root dir (must contain text/train.bin & text/val.bin)')
    p.add_argument('--neurolm_path', required=True,
                   help='Path to pretrained NeuroLM .pt checkpoint (stage-2 output)')
    p.add_argument('--debug', default=False, action='store_true',
                   help='debug=True for ProbeLabelHunterV3 (fewer tasks/files)')
    p.add_argument('--log_interval', default=10, type=int)
    # Training
    p.add_argument('--gradient_accumulation_steps', default=1, type=int)
    p.add_argument('--eeg_batch_size', default=32, type=int)
    p.add_argument('--text_batch_size', default=16, type=int)
    p.add_argument('--epochs', default=10, type=int)
    p.add_argument('--warmup_epochs', default=1, type=int)
    p.add_argument('--warmup_ratio', default=0.1, type=float)
    p.add_argument('--save_checkpoint', default=False, action='store_true',
                   help='Save a checkpoint after the final epoch')
    p.add_argument('--block_size', default=1024, type=int)
    # Optimiser
    p.add_argument('--learning_rate', default=5e-4, type=float)
    p.add_argument('--min_lr', default=5e-5, type=float)
    p.add_argument('--weight_decay', default=1e-1, type=float)
    p.add_argument('--beta1', default=0.9, type=float)
    p.add_argument('--beta2', default=0.95, type=float)
    p.add_argument('--grad_clip', default=1.0, type=float)
    p.add_argument('--decay_lr', default=True, action='store_false')
    p.add_argument('--seed', default=1337, type=int)
    p.add_argument('--compile', default=False, action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
