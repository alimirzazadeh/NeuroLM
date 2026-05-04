"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

import os
import pickle

from multiprocessing import Pool
import numpy as np
import mne


drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

standard_channels = [
    "EEG FP1-REF",
    "EEG F7-REF",
    "EEG T3-REF",
    "EEG T5-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F8-REF",
    "EEG T4-REF",
    "EEG T6-REF",
    "EEG O2-REF",
    "EEG FP1-REF",
    "EEG F3-REF",
    "EEG C3-REF",
    "EEG P3-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F4-REF",
    "EEG C4-REF",
    "EEG P4-REF",
    "EEG O2-REF",
]


def split_and_dump(params):
    fetch_folder, sub, dump_folder, label = params
    for file in os.listdir(fetch_folder):
        if sub in file:
            print("process", file)
            file_path = os.path.join(fetch_folder, file)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            try:
                if drop_channels is not None:
                    useless_chs = []
                    for ch in drop_channels:
                        if ch in raw.ch_names:
                            useless_chs.append(ch)
                    raw.drop_channels(useless_chs)
                if chOrder_standard is not None and len(chOrder_standard) == len(raw.ch_names):
                    raw.reorder_channels(chOrder_standard)
                if raw.ch_names != chOrder_standard:
                    raise Exception("channel order is wrong!")

                raw.filter(l_freq=0.1, h_freq=75.0)
                raw.notch_filter(50.0)
                raw.resample(200, n_jobs=5)

                ch_name = raw.ch_names
                raw_data = raw.get_data(units='uV')
                channeled_data = raw_data.copy()
            except:
                with open("tuab-process-error-files.txt", "a") as f:
                    f.write(file + "\n")
                continue
            for i in range(channeled_data.shape[1] // 2000):
                dump_path = os.path.join(
                    dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl"
                )
                pickle.dump(
                    {"X": channeled_data[:, i * 2000 : (i + 1) * 2000], "y": label},
                    open(dump_path, "wb"),
                )


if __name__ == "__main__":
    """
    TUAB dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

    Train/test split is loaded from external text files (one filename per line).
    Subject ID = filename.split('_')[0].
    All four source folders are searched for each subject.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="/userhome1/jiangweibang/Datasets/TUH_Abnormal/v3.0.0/edf/")
    parser.add_argument('--train_split', required=True, help='Path to tuab_train.txt')
    parser.add_argument('--test_split', required=True, help='Path to tuab_test.txt')
    parser.add_argument('--num_workers', type=int, default=24)
    args = parser.parse_args()

    root = args.root
    channel_std = "01_tcp_ar"

    # Load subject sets from split files
    def load_subjects(path):
        with open(path) as f:
            return set(line.strip().split('_')[0] for line in f if line.strip())

    train_subjects = load_subjects(args.train_split)
    test_subjects = load_subjects(args.test_split)

    print(f"Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}")

    # All source folders with their labels
    source_folders = [
        (os.path.join(root, "train", "abnormal", channel_std), 1),
        (os.path.join(root, "train", "normal",   channel_std), 0),
        (os.path.join(root, "eval",  "abnormal", channel_std), 1),
        (os.path.join(root, "eval",  "normal",   channel_std), 0),
    ]

    # Create output folders
    for split in ("train", "test"):
        os.makedirs(os.path.join(root, "processed", split), exist_ok=True)
    train_dump_folder = os.path.join(root, "processed", "train")
    test_dump_folder  = os.path.join(root, "processed", "test")

    # Build parameter list
    parameters = []
    for folder, label in source_folders:
        if not os.path.isdir(folder):
            print(f"Warning: folder not found, skipping: {folder}")
            continue
        for sub in set(item.split('_')[0] for item in os.listdir(folder) if not item.startswith('.')):
            if sub in train_subjects:
                parameters.append([folder, sub, train_dump_folder, label])
            elif sub in test_subjects:
                parameters.append([folder, sub, test_dump_folder, label])
            else:
                print(f"Subject {sub} not in any split, skipping")

    print(f"Total jobs: {len(parameters)}")

    with Pool(processes=args.num_workers) as pool:
        pool.map(split_and_dump, parameters)