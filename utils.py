"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

#from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
import math
import numpy as np
import os
from downstream_dataset import TUABLoader, TUEVLoader, TUSLLoader, HMCLoader, WorkloadLoader
from metrics import binary_metrics_fn, multiclass_metrics_fn


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def prepare_TUEV_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))

    # prepare training and test data loader
    train_dataset = TUEVLoader(
        os.path.join(
            root, "processed_train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len
    )
    test_dataset = TUEVLoader(
        os.path.join(
            root, "processed_test"), test_files, is_instruct=is_instruct, is_val=False, eeg_max_len=eeg_max_len, text_max_len=text_max_len
    )
    val_dataset = TUEVLoader(
        os.path.join(
            root, "processed_eval"), val_files, is_instruct=is_instruct, is_val=False, eeg_max_len=eeg_max_len, text_max_len=text_max_len
    )
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUAB_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUABLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=False, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files, is_instruct=is_instruct, is_val=False, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUSL_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUSLLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = TUSLLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=False, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = TUSLLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=False, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_HMC_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = HMCLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = HMCLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=False, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = HMCLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=False, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_Workload_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = WorkloadLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = WorkloadLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=False, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = WorkloadLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=False, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def get_metrics(output, target, metrics, is_binary):
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results
