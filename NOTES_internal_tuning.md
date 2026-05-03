# Internal Instruction Tuning — Notes

## Goal

Adapt NeuroLM to the internal EEG dataset (`preprocessed_eeg_v2`) via a new instruction tuning script. The model learns binary yes/no questions about EEG recordings (medications, diagnoses, EEG features).

The existing `train_instruction.py` targets public datasets (TUAB, TUEV, SEED, etc.). This script uses `ProbeLabelHunterV3` instead, which covers many binary tasks across patients.

Everything before instruction tuning (VQ training, pretraining) is unchanged — only this stage is new.

---

## Key file

`train_instruction_internal.py`

---

## Architecture context

- **3 training stages**: (1) VQ tokenizer from scratch (`train_vq.py`); (2) multi-channel EEG + OpenWebText LM pretraining (`train_pretrain.py`); (3) instruction tuning (`train_instruction_internal.py`).
- **EEG tokenization**: raw EEG → frozen `NeuralTransformer` (LaBraM architecture, submodule `model.tokenizer`) → VQ codebook indices offset by `vocab_size=50257`. Tokenizer is frozen in all downstream stages.
- **Shared vocabulary**: GPT2 vocab (50257) + EEG codebook (8192) = 58449 tokens total.
- **Sequence**: 30s × 19 channels = 570 EEG tokens + ~20 text tokens ≪ 1024 block size.
- **Causal mask**: stair-stepping — channels at same time step can attend to each other (non-causal within a second), causal across time steps and into text.
- **Loss**: answer-only cross-entropy on text (Y_text=-1 everywhere except answer token positions) + OpenWebText LM regularizer to prevent catastrophic forgetting.

---

## Checkpoint loading

- **Only `NeuroLM-B.pt` is needed** (stage-2 pretraining output). It contains both GPT2 backbone weights AND the frozen VQ encoder (`tokenizer.*` keys in state_dict).
- The VQ `.pt` (stage-1 output) is NOT passed separately — it's already embedded in the NeuroLM checkpoint.
- Pass via `--neurolm_path /path/to/NeuroLM-B.pt`.

---

## Data pipeline

- **Label system**: `ProbeLabelHunterV3` (at `/home/alimirz/2026/EEG_FM/EEG_FM/data_split_scripts/probe_label_hunter.py` on server). Returns `(N_files, N_tasks)` float tensor with values `{-1=excluded, 0=negative, 1=positive}`.
- **EEG data**: H5 files at `/orcd/compute/dinaktbi/001/2026/EEG_FM/data_EEG/preprocessed_eeg_v2`, shape `(T, C)` at 200 Hz.
- **19 channels**: `['o1','o2','t6','p4','pz','p3','t5','t3','c3','cz','c4','t4','f8','f4','fz','f3','f7','fp1','fp2']` (lowercase in H5, uppercased for `standard_1020` index lookup).
- **Train**: one sample per patient per epoch — random 30s segment, random valid task (label ≠ -1).
- **Val**: one sample per patient per epoch — fixed 30s segment (seeded per file index), fixed task (seeded per file index + 10000 offset). Same segment/task every epoch.

---

## Task → question mapping

| Prefix | Question format |
|--------|----------------|
| `med_`, `smed_` | "Is the patient taking {name}?" |
| `dis_`, `cond_`, `diag_` | "Does the patient have {name}?" |
| `feat_` | "Does the EEG show {name}?" |
| other | "Does this EEG relate to {name}?" |

Prompt format: `[SEP] Question: {question} Answer:` followed by ` Yes` or ` No` and `<|endoftext|>`.

---

## Evaluation

- **Logit-based** (no generation): `yes_logit - no_logit` score from a single forward pass with `y_text=None` (model returns last-position logits only).
- Grouped by task for batched inference (same prompt = same tensor length per task group).
- Metrics per task: `balanced_accuracy`, `ROC-AUC`, `PR-AUC`. Also logs macro mean.
- Tasks with < 2 unique labels in val are skipped.
- With one task per patient, some tasks may have very few val samples — low-sample tasks will have noisy metrics.

---

## Output locations

- **TensorBoard metrics**: `<out_dir>/runs/instruction-internal/`
- **Checkpoint** (only if `--save_checkpoint` flag set): `<out_dir>/checkpoints/instruction-internal/ckpt.pt`

---

## Run command (server, single GPU)

```bash
python train_instruction_internal.py \
    --neurolm_path /path/to/NeuroLM-B.pt \
    --out_dir /path/to/output \
    --epochs 10 \
    --eeg_batch_size 16 \
    --save_checkpoint
```

`--out_dir` must contain `text/train.bin` and `text/val.bin` (OpenWebText memmap files from `text_dataset_maker/prepare.py`).

Multi-GPU:
```bash
torchrun --nproc_per_node=4 train_instruction_internal.py \
    --neurolm_path /path/to/NeuroLM-B.pt \
    --out_dir /path/to/output \
    --eeg_batch_size 16
```

---

## Potential debugging points

1. **`ProbeLabelHunterV3` import path**: hardcoded `sys.path.insert(0, '/home/alimirz/2026/EEG_FM/EEG_FM/')` — verify this path exists on the server.

2. **H5 file shape**: script assumes `recording/data` shape `(T, C)` and `recording/ch_names` as a list. If shape is `(C, T)` or ch_names encoding differs, `_load_eeg_window` will silently produce zeros.

3. **`_CHAN_INDICES` vs `_TIME_INDICES` stride**: `_CHAN_INDICES` repeats the 19-channel list 30 times → `[ch0..ch18, ch0..ch18, ...]`. `_TIME_INDICES` is `[0×19, 1×19, ..., 29×19]`. Verify this matches what `NeuralTransformer` expects for `input_chans` / `input_time`.

4. **`model.GPT2.config.vocab_size`** used in `Y_eeg` fill value — the loaded checkpoint should have the extended vocab (50304 + 8192). If it's base 50257, the fill value will be wrong.

5. **`cosine_scheduler` `warmup_steps` kwarg**: the script passes `warmup_steps=int(args.warmup_ratio * num_steps * args.epochs)`. Check that `utils.cosine_scheduler` accepts this kwarg — the original may only take `warmup_epochs`.

6. **`model(None, None, X_text2, Y_text2)`**: verify `NeuroLM.forward` handles `x_eeg=None` (skips tokenizer branch). Check `model/model_neurolm.py` if this errors.

7. **DDP val**: val runs on master process only. Correct but slow for large val sets.
