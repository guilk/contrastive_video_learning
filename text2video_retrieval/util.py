import torch
import torch.nn as nn
import numpy as np
import os, sys
from opt import *
import random
from collections import OrderedDict,defaultdict

random.seed(43211)
np.random.seed(43211)
def load_glove(vocab, gloveFile):
    w2v = np.random.random((len(vocab) + 2, 300)).astype('float32')
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    ok = 0
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        if word in vocab:
            id = vocab[word]
            embedding = np.array([float(val) for val in splitLine[1:]])
            w2v[id + 2] = embedding
            ok += 1
    print("{}/{} words loaded!".format(ok, len(vocab)))
    return torch.from_numpy(w2v)


def l2norm(X, dim=-1, eps=1e-10):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def weights_init(m):
    #if isinstance(m, nn.Conv2d):
    #    nn.init.xavier_uniform_(m.weight.data)
    #    if m.bias is not None:
    #        nn.init.xavier_uniform_(m.bias.data)
    #elif isinstance(m, nn.Conv1d):
    #    nn.init.xavier_uniform_(m.weight.data)
    #    if m.bias is not None:
    #        nn.init.xavier_uniform_(m.bias.data)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.count == 0:
            return "%.4f" % (self.val)
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self, logfile):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()
        if os.path.isfile(logfile): # append
            self.log_fp = open(logfile, 'a')
        else: # create
            self.log_fp = open(logfile, 'w')

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)
    
    def reset(self, k):
        # create a new meter if previously not recorded
        if k in self.meters:
            self.meters[k].reset()

    def set(self, k, v):
        # create a new meter if previously not recorded
        self.meters[k] = v
    
    def pop(self, k):
        # create a new meter if previously not recorded
        self.meters.pop(k, None)

    def __str__(self):
        # Concatenate the meters in one log line
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += ','
            s += k + ':' + str(v)
        self.log_fp.write(s + '\n')
        self.log_fp.flush()
        return s
    
    def log(self):
        # Concatenate the meters in one log line
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += ','
            s += k + ':' + str(v)
        self.log_fp.write(s + '\n')
        self.log_fp.flush()


def to_tensor(array):
    return torch.from_numpy(np.array(array)).float()
    #return torch.from_numpy(np.array(array))

def to_cuda(*args):
    return [None if x is None else x.cuda() for x in args]


# Returns a dictionary of image_id -> cap
# img_to_cap: ['99884454#4']: ['a', 'man', 'on', 'a', 'moored', 'blue', 'and'...]
# max_len: 82
# id_img: id: 0~158914, img_id: 99884454#4
# => call with idx i: cap = img_to_cap[id_img[i]]
def get_cap(cap_file, max_cap_len=256):
    img_cap = {}
    id_img = {}
    max_len = 0
    print(cap_file)
    with open(cap_file, 'r', encoding='utf-8') as cap_data:
        i = 0
        for lines in cap_data.readlines():
            img_id, cap = lines.split("\t")
            img_id = os.path.splitext(img_id)[0]
            if img_id not in img_cap:
                words = cap.strip().lower().split(" ")
                img_cap[img_id] = words[0:min(len(words), max_cap_len)]
                if len(img_cap[img_id]) > max_len:
                    max_len = len(img_cap[img_id])
                id_img[i] = img_id
                i += 1
    return img_cap, id_img, max_len

# Returns a dictionary of video_id -> cap
# video_to_cap: ['videoname_startframe_endframe']: ['a', 'man', 'on', 'a', 'moored', 'blue', 'and'...]
# max_len:
# id_video: id: 0~num_events, img_id: videoname_startframe_endframe
# => call with idx i: cap = video_to_cap[id_video[i]]
def get_youcook_cap(cap_file, max_cap_len=256):
    video_cap = {}
    max_len = 0
    id_video = {}
    print(cap_file)
    with open(cap_file, 'r', encoding='utf-8') as cap_data:
        i = 0
        for lines in cap_data.readlines():
            splits = lines.split(',')
            video_id = '{}_{}_{}'.format(splits[0], splits[1], splits[2])
            cap = splits[3]

            words = cap.strip().lower().split(' ')
            video_cap[video_id] = words[0:min(len(words), max_cap_len)]
            if len(video_cap[video_id]) > max_len:
                max_len = len(video_cap[video_id])
            id_video[i] = video_id
            i += 1
    return video_cap, id_video, max_len


def frequency_map(img_cap):
    word_freq = {}
    for _, cap in img_cap.items():
        for word in cap:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
    return word_freq


def construct_vocab(word_freq, k):
    word_idx = {}
    words = sorted(list(word_freq.keys()))
    for word in words:
        freq = word_freq[word]
        if freq >= k:
            if word not in word_idx:
                word_idx[word] = len(word_idx)
    return word_idx


def get_cap_mask_len(cap, word_idx, max_len):
    idxs = np.zeros(max_len)
    # 0: padding, 1: unknown
    for i in range(len(cap)):
        if cap[i] in word_idx:
            idxs[i] = word_idx[cap[i]] + 2
        else:
            idxs[i] = 1
    mask = np.ones(max_len)
    mask[len(cap):] = 0
    length = len(cap)
    return idxs, mask, length

def img_cap_one_hot(img_cap, word_idx, max_len):
    img_cap_one_hot = {}
    for img, cap in img_cap.items():
        img_cap_one_hot[img] = get_cap_mask_len(cap, word_idx, max_len)
    return img_cap_one_hot


# if strip => 092332 # plain val/test
# if not => 092332#1~5
def get_ids(split_file, strip=False):
    ids = []
    with open(split_file, 'r', encoding='utf=8') as f:
        for id in f.readlines():
            base = os.path.basename(id.strip())
            if strip:
                ids.append(base)
            else:
                for i in range(num_caps):
                    ids.append(base + "#" + str(i))
    return ids

def get_youcook_ids(split_file):
    ids = []
    with open(split_file, 'r', encoding='utf=8') as f:
        for line in f.readlines():
            splits = line.split(',')
            video_id = '{}_{}_{}'.format(splits[0], splits[1], splits[2])
            ids.append(video_id)
    return ids

# prepare baselines for v2v/v2t/t2v/t2t retrieval
def get_youcook_pair_ids(split_file):
    ids = []
    with open(split_file, 'r', encoding='utf=8') as f:
        for line in f.readlines():
            splits = line.rstrip('\r\n').split(',')
            source_id = '{}_{}_{}'.format(splits[0], splits[1], splits[2])
            target_id = '{}_{}_{}'.format(splits[3], splits[4], splits[5])
            ids.append((source_id, target_id))
    return ids



if __name__ == '__main__':
    src_cap = '/home/liangkeg/gpu4ssd/third_hand/YouCookII/meta_data/data_splits/all_data.lst'
    src_img_cap, _, src_max_len = get_youcook_cap(src_cap)
    print(src_img_cap['XAHNVoKV1Bc_14445_14715'])
    print('The max length of the corpus: {}'.format(src_max_len))
    word_freq = frequency_map(src_img_cap)
    print(word_freq['the'], word_freq['dog'])
    vocab = construct_vocab(word_freq, 4)
    print(vocab['the'], vocab['dog'])

    # src_cap = '/home/liangkeg/gpu4ssd/third_hand/YouCookII/meta_data/data_splits/all_data.lst'
    # src_img_cap, _, src_max_len = get_youcook_cap(src_cap)
    # print(src_img_cap['XAHNVoKV1Bc_14445_14715'])
    # print('The max length of the corpus: {}'.format(src_max_len))
    # word_freq = frequency_map(src_img_cap)
    # print(word_freq['the'], word_freq['dog'])
    # vocab = construct_vocab(word_freq, 4)
    # print(vocab['the'], vocab['dog'])




    #src_img_cap, _, src_max_len = get_cap('vtt_caps.token', 999)
    #wfreq = frequency_map(src_img_cap)
    #vocab = construct_vocab(wfreq, 4)
    #print(len(vocab))
