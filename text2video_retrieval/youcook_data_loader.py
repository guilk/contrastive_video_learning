import torch.utils.data
from opt import *
from util import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import glob, random
import pickle
from tqdm import tqdm
import math

class Dataset(torch.utils.data.TensorDataset):
    def __init__(self, img_cap, ids, num_rois, vfeat_dim, vfeat_dir, tfeat_dim, tfeat_dir):
        self.img_cap = img_cap
        self.ids = ids
        self.num_samples = len(ids)
        self.num_rois = num_rois
        self.vfeat_dim = vfeat_dim
        self.vfeat_dir = vfeat_dir
        self.tfeat_dim = tfeat_dim
        self.tfeat_dir = tfeat_dir
        data = {}
        print('building dataset', len(ids))

        for index, id in enumerate(ids):
            # print('Load {}th of {} features'.format(index, len(ids)))
            feat_path = os.path.join(self.vfeat_dir, '{}.pkl'.format(id))
            feature = pickle.load(open(feat_path, 'rb'))
            if feature.ndim == 2:
                feat = feature
            else:
                feat = np.mean(feature, axis=1).astype(np.float32)
            if feat.shape[0] > self.num_rois:
                sample_rate = int(math.ceil(feat.shape[0] / self.num_rois))
                feat = feat[::sample_rate, :]
            data[id] = feat
        self.data = data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        cap, cmask, clen = self.img_cap[self.ids[idx]]
        vfeat = np.zeros((self.num_rois, self.vfeat_dim))
        vmask = np.ones(self.num_rois)

        # Load video features from memory
        feat = self.data[self.ids[idx]]

        # # Load video features from disk
        # feat_path = os.path.join(self.vfeat_dir, '{}.pkl'.format(self.ids[idx]))
        # feature = pickle.load(open(feat_path,'rb'))
        # feat = np.mean(feature, axis=1).astype(np.float32)
        # if feat.shape[0] > self.num_rois:
        #     sample_rate = int(math.ceil(feat.shape[0]/self.num_rois))
        #     feat = feat[::sample_rate,:]

        vfeat[0:min(self.num_rois, feat.shape[0]), :] = feat
        vmask[min(self.num_rois, feat.shape[0]):] = 0
        return to_tensor(cap).long(), to_tensor(cmask).bool(), \
               to_tensor(vfeat), to_tensor(vmask).bool(), \
               self.ids[idx]

class YoucookDataLoader:
    def __init__(self, params):
        self.params = params
        self.hidden_dim = params.hidden_dim

        # preprocessing
        src_img_cap, _, src_max_len = get_youcook_cap(params.src_cap)
        print('The max length of the corpus: {}'.format(src_max_len))
        vocab = construct_vocab(frequency_map(src_img_cap), 4)
        self.vocab = vocab
        if self.params.glove:
            self.w2v = load_glove(vocab, '../w2v/glove.6B.300d.txt')
        else:
            self.w2v = None

        print('Total words in vocabulary: {}'.format(len(vocab)+2))
        self.img_cap_one_hot = img_cap_one_hot(src_img_cap, vocab, src_max_len)

        # self.train_ids = get_youcook_pair_ids(os.path.join(params.split_dir, 'train.lst'))
        # self.val_ids = get_youcook_pair_ids(os.path.join(params.split_dir, 'val.lst'))
        # self.test_ids = get_youcook_pair_ids(os.path.join(params.split_dir, 'test.lst'))

        # # train/val/test ids
        self.train_ids = get_youcook_ids(os.path.join(params.split_dir, 'train.lst'))
        self.val_ids = get_youcook_ids(os.path.join(params.split_dir, 'val.lst'))
        self.test_ids = get_youcook_ids(os.path.join(params.split_dir, 'test.lst'))
        print('Number of tr_ids:{}, val_ids:{}, test_ids:{}'.format(len(self.train_ids), len(self.val_ids),
                                                                    len(self.test_ids)))

        # print(len(self.train_ids), len(self.val_ids), len(self.test_ids))
        # print(self.train_ids, self.val_ids, self.test_ids)
        # assert False

        kwargs = {'num_workers': 4, 'pin_memory':True}
        if params.is_train:
            # datasets
            self.train_data = Dataset(self.img_cap_one_hot, self.train_ids, params.num_roi, params.vfeat_dim,
                                      params.vfeat_dir, params.tfeat_dim, params.tfeat_dir)
            self.val_data = Dataset(self.img_cap_one_hot, self.val_ids, params.num_roi, params.vfeat_dim,
                                    params.vfeat_dir, params.tfeat_dim, params.tfeat_dir)
            self.test_data = Dataset(self.img_cap_one_hot, self.test_ids, params.num_roi, params.vfeat_dim,
                                     params.vfeat_dir, params.tfeat_dim, params.tfeat_dir)

            self.train_dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True,
                                               **kwargs)
            self.eval_dataloader = DataLoader(self.val_data, batch_size=self.params.batch_size, shuffle=False, **kwargs)
            self.test_dataloader = DataLoader(self.test_data, batch_size=self.params.batch_size, shuffle=False, **kwargs)
        else:
            self.test_data = Dataset(self.img_cap_one_hot, self.test_ids, params.num_roi, params.vfeat_dim,
                                     params.vfeat_dir, params.tfeat_dim, params.tfeat_dir)
            self.test_dataloader = DataLoader(self.test_data, batch_size=self.params.batch_size, shuffle=False, **kwargs)


if __name__ == '__main__':
    params = parse_arguments()
    # dl = SDataLoader(params)
    dl = YoucookDataLoader(params)
    counter = 0
    for (pos_cap, pos_cmask, pos_vfeat, pos_vmask, idxs) in tqdm(dl.train_dataloader, ascii=True, dynamic_ncols=True):
        counter += 1
    print('Total batches: {}'.format(counter))



    # feat_path = '/home/liangkeg/gpu4ssd/third_hand/YouCookII/meta_data/features/SjA7PFoZcNQ_7524_8350.pkl'
    # feature = pickle.load(open(feat_path, 'rb'))
    # print(feature.shape)
    # feature = np.mean(feature, axis=1).astype(np.float32)
    # print(feature.shape)
