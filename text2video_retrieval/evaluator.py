from util import *
from opt import *
from tqdm import tqdm
import torch
#from metrics import *
import pickle
import numpy as np
import pickle
class Evaluator:
    def __init__(self, params, dataloader):
        self.params = params
        self.dataloader = dataloader
        self.factor = params.factor


    def i2t(self, model, is_test=False):
        model.eval()
        ap_acc = 0
        if is_test:
            ids = self.dataloader.test_ids
            dataloader = self.dataloader.test_dataloader
        else:
            ids = self.dataloader.val_ids
            dataloader = self.dataloader.eval_dataloader

        print('total test len:', len(ids))
        e_vs = np.zeros((len(ids), self.factor*self.params.hidden_dim))
        e_ts = np.zeros((len(ids), self.factor*self.params.hidden_dim))
        r_1 = 0.
        r_5 = 0.
        r_10 = 0.
        count = 0
        idd = []
        # inference and get embeddings
        for cap, cmask, vfeat, vmask, id in tqdm(dataloader, ascii=True, dynamic_ncols=True):
            cap, cmask, vfeat, vmask = to_cuda(cap, cmask, vfeat, vmask)
            idd += id
            e_t, e_v, _, _ = model(cap, cmask, vfeat, vmask)
            #print(e_t[0:10,0:3])
            #print(e_v[0:10,0:3])
            e_ts[count:count+len(id)] = e_t.data.cpu().numpy()
            e_vs[count:count+len(id)] = e_v.data.cpu().numpy()
            count += len(id)

        # construct plain e_vs => e_vsp
        e_vsp = np.zeros((int(len(ids)/num_caps), self.factor*self.params.hidden_dim))
        pids = [] # 1000
        idss = [] # 5000
        count = 0
        for n, id in enumerate(idd):
            # pid = id.split('#')[0]
            pid = id
            idss.append(pid)
            if pid not in pids:
                pids += [pid]
                e_vsp[count] = e_vs[n]
                count += 1

        len_pids = len(pids)
        pids = np.array(pids)
        idss = np.array(idss)
        sims = np.matmul(e_vsp, e_ts.T)
        save_data = {}
        for n, id in enumerate(pids.tolist()):
            pid = id
            top_10_img_idx = (-np.asarray(sims[n])).argsort()[:10]
            top_pids = idss[top_10_img_idx]
            save_data[pid] = top_pids
            if pid == top_pids[0]:
                r_1 += 1
                r_5 += 1
                r_10 += 1
            elif pid in top_pids[1:5]:
                r_5 += 1
                r_10 += 1
            elif pid in top_pids[5:10]:
                r_10 += 1

        pickle.dump(save_data, open('./i2t_predictions.pkl', 'wb'))
        return r_1 / len_pids, r_5 / len_pids, r_10 / len_pids




    def t2i(self, model, is_test=False):
        model.eval()
        if is_test:
            ids = self.dataloader.test_ids
            dataloader = self.dataloader.test_dataloader
        else:
            ids = self.dataloader.val_ids
            dataloader = self.dataloader.eval_dataloader

        print('total test len:', len(ids))
        e_vs = np.zeros((len(ids), self.factor*self.params.hidden_dim))
        e_ts = np.zeros((len(ids), self.factor*self.params.hidden_dim))
        r_1 = 0.
        r_5 = 0.
        r_10 = 0.
        ap_acc= 0.
        count = 0
        idd = []
        # inference and get embeddings
        for cap, cmask, vfeat, vmask, id in tqdm(dataloader, ascii=True, dynamic_ncols=True):
            cap, cmask, vfeat, vmask = to_cuda(cap, cmask, vfeat, vmask)
            idd += id
            e_t, e_v, _, _ = model(cap, cmask, vfeat, vmask)
            e_ts[count:count+len(id)] = e_t.data.cpu().numpy()
            e_vs[count:count+len(id)] = e_v.data.cpu().numpy()
            count += len(id)
        # construct plain e_vs => e_vsp
        e_vsp = np.zeros((int(len(ids)/num_caps), self.factor*self.params.hidden_dim))
        pids = []
        count = 0
        for n, id in enumerate(idd):
            # pid = id.split('#')[0]
            pid = id
            if pid not in pids:
                pids += [pid]
                e_vsp[count] = e_vs[n]
                count += 1
        # debug
        save_data = {}

        pids = np.array(pids)
        count = 0
        sims = np.matmul(e_ts, e_vsp.T)
        for n, id in enumerate(idd):
            # pid = id.split('#')[0]
            pid = id
            top_10_img_idx = (-np.asarray(sims[n])).argsort()[:10]
            top_pids = pids[top_10_img_idx]
            save_data[pid] = top_pids
            # print(pid, top_pids)
            if pid == top_pids[0]:
                r_1 += 1
                r_5 += 1
                r_10 += 1
            elif pid in top_pids[1:5]:
                r_5 += 1
                r_10 += 1
            elif pid in top_pids[5:10]:
                r_10 += 1
        pickle.dump(save_data, open('./t2i_predictions.pkl','wb'))
        return r_1 / len(ids), r_5 / len(ids), r_10 / len(ids)