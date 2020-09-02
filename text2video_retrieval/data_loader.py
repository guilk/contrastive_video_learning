import torch.utils.data
from opt import *
from util import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import glob, random

class Dataset(torch.utils.data.TensorDataset):
    def __init__(self, img_cap, ids, num_roi, vfeat_dim, vfeat_dir, tfeat_dim, tfeat_dir):
        self.img_cap = img_cap
        self.ids = ids
        self.num_of_samples = len(ids)
        self.num_roi = num_roi
        self.vfeat_dim = vfeat_dim
        self.vfeat_dir = vfeat_dir
        self.tfeat_dim = tfeat_dim
        self.tfeat_dir = tfeat_dir
        data = {}
        print('building dataset', len(ids))
        for id in ids:
            idd = id.split('#')[0]
            data[idd] = np.load(os.path.join(self.vfeat_dir, idd + '.npy')).astype('float32')
        self.data = data
        

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        cap, cmask, clen = self.img_cap[self.ids[idx]]
        vfeat = np.zeros((self.num_roi, self.vfeat_dim))
        vmask = np.ones(self.num_roi)

        #feat = np.load(os.path.join(self.vfeat_dir, self.ids[idx].split("#")[0] + '.npy'))
        feat = self.data[self.ids[idx].split("#")[0]]


        vfeat[0:min(self.num_roi, feat.shape[0]),:] = feat
        vmask[min(self.num_roi, feat.shape[0]):] = 0
        return to_tensor(cap).long(), to_tensor(cmask).bool(), \
               to_tensor(vfeat), to_tensor(vmask).bool(), \
               self.ids[idx]

class SDataLoader:
    def __init__(self, params):
        self.params = params
        self.hidden_dim = params.hidden_dim

        # preprocessing
        src_img_cap, _, src_max_len = get_cap(params.src_cap)
        tgt_img_cap, _, tgt_max_len = get_cap(params.tgt_cap)
        print("Src max len: {}, Tgt max len:{}".format(src_max_len, tgt_max_len))
        vocab = construct_vocab(frequency_map(src_img_cap), 4)
        self.vocab = vocab
        if self.params.glove:
            self.w2v = load_glove(vocab, 'w2v/glove.6B.300d.txt') 
        else:
            self.w2v = None

        print("Total words in vocabulary: {}".format(len(vocab) + 2))
        self.img_cap_one_hot = img_cap_one_hot(tgt_img_cap, vocab, tgt_max_len)

        # train/val/test ids
        self.train_ids = get_ids(os.path.join(params.split_dir, 'train.lst'))
        self.val_ids = get_ids(os.path.join(params.split_dir, 'val.lst'))
        self.test_ids = get_ids(os.path.join(params.split_dir, 'test.lst'))
        print('Number of tr_ids:{}, val_ids:{}, test_ids:{}'.format(len(self.train_ids), len(self.val_ids), len(self.test_ids)))

        kwargs = {'num_workers': 4, 'pin_memory': True}
        if params.is_train:
            # datasets
            self.train_data = Dataset(self.img_cap_one_hot, self.train_ids, params.num_roi, params.vfeat_dim, params.vfeat_dir, params.tfeat_dim, params.tfeat_dir)
            self.val_data   = Dataset(self.img_cap_one_hot, self.val_ids, params.num_roi, params.vfeat_dim, params.vfeat_dir, params.tfeat_dim, params.tfeat_dir)
            # dataloaders                                                                
            self.train_dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, **kwargs)
            self.eval_dataloader = DataLoader(self.val_data, batch_size=self.params.batch_size, shuffle=False, **kwargs)
        else:
            self.test_data  = Dataset(self.img_cap_one_hot, self.test_ids, params.num_roi, params.vfeat_dim, params.vfeat_dir, params.tfeat_dim, params.tfeat_dir)
            self.test_dataloader = DataLoader(self.test_data, batch_size=self.params.batch_size, shuffle=False, **kwargs)





