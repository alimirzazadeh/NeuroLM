"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

import mne
import numpy as np
import os
import pickle
from tqdm import tqdm
import pandas as pd

"""
https://github.com/Abhishaike/EEG_Event_Classification
"""

drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

symbols_tusl = {'bckg': 0, 'seiz': 1, 'slow': 2}

def BuildEvents(signals, times, EventData):
    [numEvents, z] = EventData.shape  # numEvents is equal to # of rows of the .rec file
    fs = 200.0
    [numChan, numPoints] = signals.shape

    features = np.zeros([numEvents, numChan, int(fs) * 10])
    labels = np.zeros([numEvents, 1])
    for i, (_, row) in enumerate(EventData.iterrows()):
        start = np.where((times) >= row[0])[0][0]
        end = np.where((times) >= row[1])[0][0]
        features[i, :] = signals[:, start:end]
        labels[i, :] = symbols_tusl[row[2]]
    return [features, labels]


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    if drop_channels is not None:
        useless_chs = []
        for ch in drop_channels:
            if ch in Rawdata.ch_names:
                useless_chs.append(ch)
        Rawdata.drop_channels(useless_chs)
    if chOrder_standard is not None and len(chOrder_standard) == len(Rawdata.ch_names):
        Rawdata.reorder_channels(chOrder_standard)
    if Rawdata.ch_names != chOrder_standard:
        raise ValueError

    Rawdata.filter(l_freq=0.1, h_freq=75.0)
    Rawdata.notch_filter(60.0)
    Rawdata.resample(200, n_jobs=5)

    _, times = Rawdata[:]
    signals = Rawdata.get_data(units='uV')
    labelFile = fileName[0:-3] + "tse_agg"
    labelData = pd.read_csv(labelFile, header=None, sep=' ', skiprows=2)
    Rawdata.close()
    return [signals, times, labelData, Rawdata]


def load_up_objects(fileList, Features, Labels, OutDir):
    for fname in fileList:
        print("\t%s" % fname)
        try:
            [signals, times, labelData, Rawdata] = readEDF(fname)
        except (ValueError, KeyError):
            print("something funky happened in " + fname)
            continue
        signals, labels = BuildEvents(signals, times, labelData)

        for idx, (signal, label) in enumerate(
            zip(signals, labels)
        ):
            sample = {
                "X": signal,
                "ch_names": [name.split(' ')[-1].split('-')[0] for name in chOrder_standard],
                "y": label,
            }
            print(signal.shape)

            save_pickle(
                sample,
                os.path.join(
                    OutDir, fname.split(".")[0].split("/")[-1] + "-" + str(idx) + ".pkl"
                ),
            )

    return Features, Labels


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


#root = "/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUEV/data/v2.0.1/edf"
root = "/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUSL/data/v2.0.1/edf"
out_dir = "/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUSL/neurolm"
#root = "/D_data/yansen/tuh_data/tuh_eeg_slowing/edf"
#out_dir = '../teamdrive/TUSL'
train_out_dir = os.path.join(out_dir, "train")
eval_out_dir = os.path.join(out_dir, "eval")
test_out_dir = os.path.join(out_dir, "test")
if not os.path.exists(train_out_dir):
    os.makedirs(train_out_dir)
if not os.path.exists(eval_out_dir):
    os.makedirs(eval_out_dir)
if not os.path.exists(test_out_dir):
    os.makedirs(test_out_dir)

edf_files = []
for dirName, subdirList, fileList in os.walk(root):
    for fname in fileList:
        if fname[-4:] == ".edf":
            edf_files.append(os.path.join(dirName, fname))

train_files = edf_files[:int(len(edf_files) * 0.6)]
test_files = edf_files[int(len(edf_files) * 0.6):int(len(edf_files) * 0.8)]
eval_files = edf_files[int(len(edf_files) * 0.8):]

fs = 200
TrainFeatures = np.empty(
    (0, 23, fs * 10)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
TrainLabels = np.empty([0, 1])
load_up_objects(
    train_files, TrainFeatures, TrainLabels, train_out_dir
)

fs = 200
EvalFeatures = np.empty(
    (0, 23, fs * 10)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
EvalLabels = np.empty([0, 1])
load_up_objects(
    eval_files, EvalFeatures, EvalLabels, eval_out_dir
)

fs = 200
TestFeatures = np.empty(
    (0, 23, fs * 10)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
TestLabels = np.empty([0, 1])
load_up_objects(
    test_files, TestFeatures, TestLabels, test_out_dir
)
