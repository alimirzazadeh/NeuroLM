import os
import sys
import time
import argparse
from contextlib import nullcontext

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter
import tiktoken

import pandas as pd

from model.model_neurolm import NeuroLM
from model.model import GPTConfig
from dataset import standard_1020
from utils import cosine_scheduler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EEG_DATA_DIR       = '/orcd/compute/dinaktbi/001/2026/EEG_FM/preprocessed_eeg_v2'
_TAMING_DATA       = '/home/alimirz/2026/EEG_FM/EEG_FM/taming-transformers/data'
_BAD_H5_FILES      = '/home/alimirz/2026/EEG_FM/EEG_FM/data_split_scripts/bad_h5_files.txt'
_DATA_SPLIT_SCRIPTS = '/home/alimirz/2026/EEG_FM/EEG_FM/data_split_scripts'

# ProbeLabelHunterV2 lives outside this repo — add its directory to the path
if _DATA_SPLIT_SCRIPTS not in sys.path:
    sys.path.insert(0, _DATA_SPLIT_SCRIPTS)
from probe_label_hunter import ProbeLabelHunterV2

# 19-channel order used by preprocessed_eeg_v2 (lowercase, as stored in H5)
CHANNEL_ORDER = ['o1', 'o2', 't6', 'p4', 'pz', 'p3', 't5', 't3',
                 'c3', 'cz', 'c4', 't4', 'f8', 'f4', 'fz', 'f3', 'f7', 'fp1', 'fp2']
CHANNEL_ORDER_UPPER = [ch.upper() for ch in CHANNEL_ORDER]

FS = 200
WINDOW_SECONDS = 4
WINDOW_SAMPLES = WINDOW_SECONDS * FS   # 6000
NUM_CHANS = len(CHANNEL_ORDER)         # 19
NUM_TIME = WINDOW_SECONDS              # 4
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
# Label loading via ProbeLabelHunterV2 (propensity-score matched per task)
# ---------------------------------------------------------------------------

def load_label_data(eeg_data_dir=EEG_DATA_DIR, debug=False):
    """
    Use ProbeLabelHunterV2 to get propensity-score-matched labels per task.
    Labels are:  1 = positive,  0 = matched negative,  -1 = excluded.
    Returns: train_filenames, val_filenames, downstream_tasks, get_labels(filenames)
    """
    hunter = ProbeLabelHunterV2(eeg_data_dir, size='large', debug=debug)

    downstream_tasks = list(hunter.downstream_tasks)
    if debug:
        downstream_tasks = ['feat_sleep', 'feat_generalized slowing',
                            'feat_posterior dominant rhythm', 'feat_diffuse beta']

    sessions_dict = hunter.sessions_patient_dict  # h5 filename → composite key

    def get_labels(filenames):
        keys = [sessions_dict[f] for f in filenames]
        block = hunter.matched_dfs.loc[keys, downstream_tasks]
        arr = block.apply(pd.to_numeric, errors='coerce').fillna(-1).astype(np.float32).values
        return torch.tensor(arr)

    return hunter.train_filenames, hunter.val_filenames, downstream_tasks, get_labels


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

        # Precompute full-text token tensors for both labels, and prompt-only lengths
        self._full_text_tokens: dict[tuple, torch.Tensor] = {}
        self.prompt_tokens: dict[str, torch.Tensor] = {}
        for task in self.tasks:
            prompt_ids = enc.encode(build_prompt(task))
            self.prompt_tokens[task] = torch.tensor(prompt_ids[:TEXT_MAX_LEN], dtype=torch.long)
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

        # Global z-score over all samples and channels — matches pretraining std_norm
        mean = ordered.mean()
        std = ordered.std() + 1e-8
        ordered = (ordered - mean) / std

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
            X_text, Y_text = self._build_train_text(task_name, label)
            gpt_mask = self._build_gpt_mask(TEXT_MAX_LEN)
            return X_eeg, X_text, Y_text, input_chans, input_time, eeg_mask, gpt_mask, task_idx, label

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
            return X_eeg, X_text, Y_text, input_chans, input_time, eeg_mask, gpt_mask, task_idx, label


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
def evaluate(model, val_loader, get_batch, writer, global_step: int, vocab_size: int):
    model.eval()

    total_instr_loss = 0.0
    total_text_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        if batch is None:
            continue
        X_eeg, X_text, Y_text, input_chans, input_time, eeg_mask, gpt_mask, _task_idx, _label = batch
        X_eeg = X_eeg.float().to(device, non_blocking=True)
        X_text = X_text.to(device, non_blocking=True)
        Y_text = Y_text.to(device, non_blocking=True)
        input_chans = input_chans.to(device, non_blocking=True)
        input_time = input_time.to(device, non_blocking=True)
        eeg_mask = eeg_mask.to(device, non_blocking=True)
        gpt_mask = gpt_mask.to(device, non_blocking=True)

        Y_eeg = torch.full((X_eeg.size(0), X_eeg.size(1)),
                           fill_value=-1 - vocab_size, device=device)

        X_text2, Y_text2 = get_batch('val')

        with ctx:
            _, log1, _ = model(X_eeg, Y_eeg, X_text, Y_text,
                               input_chans, input_time, eeg_mask,
                               eeg_text_mask=gpt_mask)
            _, log2, _ = model(None, None, X_text2, Y_text2)

        total_instr_loss += log1['val/loss']
        total_text_loss += log2['val/loss']
        n_batches += 1

    if n_batches == 0:
        model.train()
        return {}

    instr_loss = total_instr_loss / n_batches
    text_loss = total_text_loss / n_batches
    print(f"  [val]  instr={instr_loss:.4f}  text={text_loss:.4f}")

    if writer is not None:
        writer.add_scalar('val/instruction_loss', instr_loss, global_step)
        writer.add_scalar('val/text_loss', text_loss, global_step)
        writer.add_scalar('val/total_loss', instr_loss + text_loss, global_step)

    model.train()
    return {'instr_loss': instr_loss, 'text_loss': text_loss}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    # Timestamped experiment directory — unique per run, never overwritten
    prefix = f'{args.name}_' if args.name else ''
    exp_dir = os.path.join(args.out_dir, f'{prefix}exp')
    checkpoint_out_dir = os.path.join(exp_dir, 'checkpoints')
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)
        print(f"Experiment dir: {exp_dir}")

    writer = None
    if master_process:
        writer = SummaryWriter(os.path.join(exp_dir, 'runs'))

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

    # Labels and file splits — via ProbeLabelHunterV2 (propensity-score matched)
    if master_process:
        print("Loading labels via ProbeLabelHunterV2 ...")
    train_filenames, val_filenames, downstream_tasks, get_labels = load_label_data(
        eeg_data_dir=EEG_DATA_DIR, debug=args.debug,
    )
    if master_process:
        print(f"{len(downstream_tasks)} tasks: {downstream_tasks[:5]} ...")

    train_labels = get_labels(train_filenames)
    val_labels   = get_labels(val_filenames)

    train_dataset = SafeDataset(InternalInstructDataset(
        train_filenames, train_labels, downstream_tasks,
        eeg_data_dir=EEG_DATA_DIR, mode='train', seed=args.seed,
    ))
    val_dataset = SafeDataset(InternalInstructDataset(
        val_filenames, val_labels, downstream_tasks,
        eeg_data_dir=EEG_DATA_DIR, mode='val', seed=args.seed,
    ))
    if master_process:
        print(f"Train files: {len(train_dataset)}  |  Val files: {len(val_dataset)}")

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

    val_loader = DataLoader(val_dataset, batch_size=args.eeg_batch_size,
                            shuffle=False, num_workers=4,
                            pin_memory=True, drop_last=False,
                            collate_fn=safe_collate_fn)

    # Model init
    iter_num = 0
    n_layer, n_head, n_embd = 12, 12, 768
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                      block_size=args.block_size, bias=False,
                      vocab_size=50257, dropout=0.0)

    _resume_candidates = ['ckpt_final.pt', 'ckpt_b.pt', 'ckpt_a.pt']
    resume_path = next(
        (os.path.join(checkpoint_out_dir, n) for n in _resume_candidates
         if os.path.exists(os.path.join(checkpoint_out_dir, n))), None
    )
    if resume_path:
        if master_process:
            print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint['model_args'][k]
        gptconf = GPTConfig(**model_args)
        model = NeuroLM(gptconf, init_from='scratch')
        _load_state_dict(model, checkpoint['model'])
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1
        if master_process:
            print(f"Resuming from epoch {checkpoint['epoch']+1}")
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
    if resume_path:
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

    _enc = tiktoken.get_encoding("gpt2")

    X_text2, Y_text2 = get_batch('train')
    t0 = time.time()

    n_tasks = len(downstream_tasks)
    task_pos_counts   = np.zeros(n_tasks, dtype=np.int64)
    task_total_counts = np.zeros(n_tasks, dtype=np.int64)

    for epoch in range(start_epoch, args.epochs):
        if ddp:
            train_loader.sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue

            lr = lr_schedule[iter_num] if args.decay_lr else args.learning_rate
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            X_eeg, X_text, Y_text, input_chans, input_time, eeg_mask, gpt_mask, task_idx_batch, label_batch = batch

            if master_process:
                ti_np = task_idx_batch.numpy()
                lb_np = label_batch.numpy()
                np.add.at(task_total_counts, ti_np, 1)
                np.add.at(task_pos_counts,   ti_np, (lb_np == 1).astype(np.int64))
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
                if args.verbose:
                    sample_ids = X_text[0].cpu().tolist()
                    sample_ids = [t for t in sample_ids if t != 50256]  # strip EOT padding
                    print(f"  [sample] {_enc.decode(sample_ids)}")
                writer.add_scalar('train/total_loss', instr_loss + text_loss, iter_num)
                writer.add_scalar('train/instruction_loss', instr_loss, iter_num)
                writer.add_scalar('train/text_loss', text_loss, iter_num)
                writer.add_scalar('train/instruction_accuracy', instr_acc, iter_num)
                writer.add_scalar('train/lr', lr, iter_num)
                t0 = t1

            iter_num += 1

        # Per-task positive-rate summary for this epoch
        if master_process:
            print(f"[epoch {epoch}] task sampling stats (master process only):")
            for i, task in enumerate(downstream_tasks):
                total = task_total_counts[i]
                if total > 0:
                    pct = 100.0 * task_pos_counts[i] / total
                    print(f"  {task}: {pct:.1f}% pos  ({task_pos_counts[i]}/{total})")
            task_pos_counts[:]   = 0
            task_total_counts[:] = 0

        # Checkpointing: every other epoch, alternating between two slots to
        # avoid accumulation; always save a final checkpoint at the last epoch.
        is_final = (epoch == args.epochs - 1)
        if master_process and (epoch % 2 == 0 or is_final):
            ckpt = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'epoch': epoch,
            }
            if is_final:
                path = os.path.join(checkpoint_out_dir, 'ckpt_final.pt')
            else:
                slot = 'ckpt_a.pt' if (epoch // 2) % 2 == 0 else 'ckpt_b.pt'
                path = os.path.join(checkpoint_out_dir, slot)
            torch.save(ckpt, path)
            print(f"Checkpoint saved to {path}")

        # Validation (master only — avoids DDP sync complexity)
        if master_process:
            evaluate(raw_model, val_loader, get_batch, writer,
                     global_step=iter_num, vocab_size=raw_model.GPT2.config.vocab_size)

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
                   help='Use only 4 feat_ tasks (faster debugging)')
    p.add_argument('--name', default='', type=str,
                   help='Optional prefix for the experiment folder name')
    p.add_argument('--log_interval', default=10, type=int)
    p.add_argument('--verbose', default=False, action='store_true',
                   help='Print a decoded sample question at each log interval')
    # Training
    p.add_argument('--gradient_accumulation_steps', default=1, type=int)
    p.add_argument('--eeg_batch_size', default=64, type=int)
    p.add_argument('--text_batch_size', default=16, type=int)
    p.add_argument('--epochs', default=100, type=int)
    p.add_argument('--warmup_epochs', default=5, type=int)
    p.add_argument('--warmup_ratio', default=0.1, type=float)
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
