"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

from torch.utils.data import Dataset
from pathlib import Path
import h5py
import bisect
import torch
from einops import rearrange
import tiktoken
import numpy as np
import pickle
import os
from dataset import standard_1020


def get_chans(ch_names):
    chans = []
    for ch_name in ch_names:
        chans.append(standard_1020.index(ch_name))
    return chans


class SEEDDataset(Dataset):
    """读取单个hdf5文件，仅使用范式中途采集的数据，标签内包含范式标签与被试性别，如有需要可以继续往字典中添加"""
    def __init__(self, file_path: Path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1, 
                 trial_start_percentage: float=0, trial_end_percentage: float=1, subject_start_percentage: float=0, subject_end_percentage: float=1, 
                 is_instruct: bool=False, is_val: bool=False, eeg_max_len=-1, text_max_len=-1):
        '''
        从路径file_path中提取数据集。

        :param Path file_path: 目标数据路径
        :param int window_size: 单个样本长度
        :param int stride_size: 两个相邻样本间隔
        :param float start_percentage: 数据集中，每个采纳的trial内首个样本在此trial的样本中的百分比索引（包括）。
        :param float end_percentage: 数据集中，每个采纳的trial内末尾样本在此trial的样本中的百分比索引（不包括）。
        :param float trial_start_percentage: 数据集中，采纳的首个trial在此被试的所有trial中的百分比索引（包括）。
        :param float trial_end_percentage: 数据集中，采纳的末个trial在此被试的所有trial中的百分比索引（不包括）。
        :param float subject_start_percentage: 数据集中，采纳的首个被试的百分比索引（包括）。
        :param float subject_end_percentage: 数据集中，采纳的末个被试的百分比索引（不包括）。
        
        比如，数据文件总共10个被试，每个被试有15个trial，每个trial提供100个样本时。取参数为0.2, 0.8, 0.34, 0.67, 0.2, 0.8时，数据集会包括下标为[2, 8)的被试，每个被试的下标为[5, 10)的trial中，每个trial下标为[20, 80)的样本。
        '''
        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage
        self.__trial_start_percentage = trial_start_percentage
        self.__trial_end_percentage = trial_end_percentage
        self.__subject_start_percentage = subject_start_percentage
        self.__subject_end_percentage = subject_end_percentage
        self.__is_instruct = is_instruct
        self.__is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        self.__file = None
        self.__length = None
        self.__feature_size = None

        self.__subjects = []
        self.__global_idxes = [] # 从第几个样本开始是哪个被试
        self.__local_idxess = [] # 从这个被试的第几个样本开始是哪个trial
        self.__trial_start_idxess = [] # trial开始索引
        self.__genders = []
        self.__labelss = []

        self.__rsFreq = None

        self.__seed_label = {
            'H': 0,
            'N': 1,
            'S': 2
        }
        
        self.__init_dataset()

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.__text = {
                0: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: Positive <|endoftext|>')),
                2: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: Negative <|endoftext|>')),
                1: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: Neutral <|endoftext|>'))
            }
            self.__prompt = torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer:'))

    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.__subjects = [i for i in self.__file]

        global_idx = 0
        subject_start_id = int(len(self.__subjects) * self.__subject_start_percentage) # 包括在数据集中的被试开始id
        subject_end_id = int(len(self.__subjects) * self.__subject_end_percentage - 1) # 包括在数据集中的被试结束id
        for subject_id, subject in enumerate(self.__subjects):
            self.__global_idxes.append(global_idx)
            #self.__genders.append(self.__file[subject].attrs['gender'])
            self.__labelss.append(self.__file[subject].attrs['label'])
            self.__rsFreq = self.__file[subject]['eeg'].attrs['rsFreq']

            local_idxes = [] # 当前trial的第一个样本在数据集中的样本索引
            trial_start_idxes = [] # 当前trial在原始数据中的开始位置索引
            trial_starts = self.__file[subject].attrs['trialStart']
            trial_ends = self.__file[subject].attrs['trialEnd']
            local_idx = 0
            if subject_id >= subject_start_id and subject_id <= subject_end_id:
                trial_start_id = int(len(trial_starts) * self.__trial_start_percentage)  # 该被试包括在数据集中的trial开始id
                trial_end_id = int(len(trial_starts) * self.__trial_end_percentage - 1)  # 该被试包括在数据集中的trial结束id
                for trial_id, (trial_start, trial_end) in enumerate(zip(trial_starts, trial_ends)):
                    local_idxes.append(local_idx)

                    if trial_id >= trial_start_id and trial_id <= trial_end_id:
                        trial_len = (trial_end - trial_start + 1) * self.__rsFreq
                        trial_sample_num = (trial_len-self.__window_size) // self.__stride_size + 1
                        start_idx = int(trial_sample_num * self.__start_percentage) * self.__stride_size + trial_start * self.__rsFreq
                        end_idx = int(trial_sample_num * self.__end_percentage - 1) * self.__stride_size + trial_start * self.__rsFreq

                        trial_start_idxes.append(start_idx)
                        local_idx += (end_idx - start_idx) // self.__stride_size + 1
                    else:
                        trial_start_idxes.append(0)

            self.__local_idxess.append(local_idxes)
            self.__trial_start_idxess.append(trial_start_idxes)

            global_idx += local_idx

        self.__length = global_idx

        self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self):
        return self.__feature_size
    
    @property
    def rsfreq(self):
        return self.__rsFreq

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        # 先确认样本属于哪个被试，再确认样本属于哪个trial
        subject_id = bisect.bisect(self.__global_idxes, idx) - 1
        trial_id = bisect.bisect(self.__local_idxess[subject_id], idx-self.__global_idxes[subject_id]) - 1
        item_start_idx = (idx - self.__global_idxes[subject_id] - self.__local_idxess[subject_id][trial_id]) * self.__stride_size + self.__trial_start_idxess[subject_id][trial_id]
        
        X = self.__file[self.__subjects[subject_id]]['eeg'][:, item_start_idx:item_start_idx+self.__window_size]
        Y = self.__seed_label[self.__labelss[subject_id][trial_id]]

        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = self.get_ch_names()
        input_chans = list(self.get_ch_names()) * time

        if not self.__is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.__is_val:
            text = self.__prompt
        else:
            text = self.__text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.__is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.__prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()

    def free(self) -> None: 
        # TODO 临时方案，目标：减少文件打开次数。查一下flush
        if self.__file:
            self.__file.close()
            self.__file = None
    
    def get_ch_names(self):
        return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']


class TUABLoader(Dataset):
    # abnormal: 1
    # normal: 0
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        self.ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                1: torch.IntTensor([50257] + encode('Question: Is this EEG segment abnormal? Answer: Yes <|endoftext|>')),
                0: torch.IntTensor([50257] + encode('Question: Is this EEG segment abnormal? Answer: No <|endoftext|>'))
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Is this EEG segment abnormal? Answer:'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        Y = sample["y"]

        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = self.ch_names
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
    

class TUEVLoader(Dataset):
    # spsw: spike and slow wave
    # gped: generalized periodic epileptiform discharge
    # pled: periodic lateralized epileptiform dischage
    # eyem: eye movement
    # artf: artifact
    # bckg: background
    # 1: spsw
    # 2: gped
    # 3: pled
    # 4: eyem
    # 5: artf
    # 6: bckg
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        self.ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                0: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (A) <|endoftext|>')),
                1: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (B) <|endoftext|>')),
                2: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (C) <|endoftext|>')),
                3: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (D) <|endoftext|>')),
                4: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (E) <|endoftext|>')),
                5: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (F) <|endoftext|>')),
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: ('))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        Y = int(sample["label"][0] - 1)
        
        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = self.ch_names
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0) - 1
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
    

class TUSLLoader(Dataset):
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        self.ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                0: torch.IntTensor([50257] + encode('Question: Which type does this EEG segment belong to? Options: (A) background. (B) seizure. (C) slowing. Answer: (A) <|endoftext|>')),
                1: torch.IntTensor([50257] + encode('Question: Which type does this EEG segment belong to? Options: (A) background. (B) seizure. (C) slowing. Answer: (B) <|endoftext|>')),
                2: torch.IntTensor([50257] + encode('Question: Which type does this EEG segment belong to? Options: (A) background. (B) seizure. (C) slowing. Answer: (C) <|endoftext|>'))
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Which type does this EEG segment belong to? Options: (A) background. (B) seizure. (C) slowing. Answer: ('))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        Y = int(np.asarray(sample["y"]).flat[0])

        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = self.ch_names
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()


class HMCLoader(Dataset):
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                0: torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (A) <|endoftext|>')),
                1: torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (B) <|endoftext|>')),
                2: torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (C) <|endoftext|>')),
                3: torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (D) <|endoftext|>')),
                4: torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (E) <|endoftext|>')),
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: ('))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        Y = int(np.asarray(sample["y"]).flat[0])

        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = sample["ch_names"]
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0) - 1
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()


class WorkloadLoader(Dataset):
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                1: torch.IntTensor([50257] + encode('Question: Is this EEG segment of high workload? Answer: Yes <|endoftext|>')),
                0: torch.IntTensor([50257] + encode('Question: Is this EEG segment of high workload? Answer: No <|endoftext|>')),
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Is this EEG segment of high workload? Answer:'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        Y = int(np.asarray(sample["y"]).flat[0])

        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = sample["ch_names"]
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
