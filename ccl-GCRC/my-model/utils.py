# -*- coding: utf-8 -*-

import os
import pickle
import re, json
import numpy as np
from tqdm import tqdm
from typing import List, Dict

def dump_file(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f)

def load_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def get_dir_files(dirname):
    L = []
    for root, dirs, files in os.walk(dirname):
        for file in files:
            L.append(os.path.join(root, file))
    return L

def padding(sequence, pads=0, max_len=None, dtype='int32'):
    v_length = [len(x) for x in sequence]  # every sequence length
    seq_max_len = max(v_length)
    if (max_len is None) or (max_len > seq_max_len):
        max_len = seq_max_len
    x = (np.ones((len(sequence), max_len)) * pads).astype(dtype)
    for idx, s in enumerate(sequence):
        trunc = s[:max_len]
        x[idx, :len(trunc)] = trunc
    return x
