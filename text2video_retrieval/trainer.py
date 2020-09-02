import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer as timer
from util import *
import os, sys, time
from tqdm import tqdm
from opt import *
from loss import TripletLoss

class Trainer:
    def __init__(self, params, model, dataloader, evaluator):
        self.params = params
        self.dataloader = dataloader
        self.evaluator = evaluator
        if self.params.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.params.lr, weight_decay=self.params.wd)
        elif self.params.optimizer == 'Adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), lr=self.params.lr, weight_decay=self.params.wd)
        else:
            print('Invalid optimizer, quit')
            exit(0)
        self.optimizer = optimizer
        self.loss_function = TripletLoss(margin=params.margin)
        self.logger = LogCollector(os.path.join(self.params.model_path, 'tr_log'))

    def train(self, model, model_epoch, best_prev):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        for epoch in range(model_epoch, self.params.num_epochs):
            model.train()
            count = 1
            # adjust step lr
            if epoch != 0 and epoch % self.params.step_size == 0:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / self.params.gamma
                print('change lr to: {}'.format(optim_state['param_groups'][0]['lr']))
                self.optimizer.load_state_dict(optim_state)
            # mini-batch training
            start = time.time()
            for (pos_cap, pos_cmask, pos_vfeat, pos_vmask, idxs) in tqdm(self.dataloader.train_dataloader, ascii=True, dynamic_ncols=True):
                data_time.update(time.time() - start)
                bs = pos_cap.size(0)

                self.optimizer.zero_grad()
                pos_cap, pos_cmask, pos_vfeat, pos_vmask = to_cuda(pos_cap, pos_cmask, pos_vfeat, pos_vmask)
                t_emb, v_emb, _, _ = model(pos_cap, pos_cmask, pos_vfeat, pos_vmask)
                loss = self.loss_function(t_emb, v_emb)

                # optimization
                loss.backward()
                if self.params.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.params.clip)
                self.optimizer.step()
                
                # update log
                self.logger.update('train_loss', loss.data.cpu().numpy(), bs)
                batch_time.update(time.time() - start)
                start = time.time()

                # record log
                if count % 60 == 0:
                    self.logger.set('epoch', epoch)
                    self.logger.set('iter', count)
                    tqdm.write("{}, data time: {:.4f}, batch time: {:.4f}".format(
                                str(self.logger), data_time.avg, batch_time.avg))
                count += 1

            # Calculate r@k after validate_every epoch
            if (epoch + 1) % self.params.validate_every == 0:
                r_1, r_5, r_10 = self.evaluator.t2i(model, is_test=False)
                r_1_test, r_5_test, r_10_test = self.evaluator.t2i(model, is_test=True)
                self.logger.set('epoch', epoch+1)
                self.logger.set('r@1', r_1)
                self.logger.set('r@5', r_5)
                self.logger.set('r@10', r_10)
                self.logger.set('r@1_test', r_1_test)
                self.logger.set('r@5_test', r_5_test)
                self.logger.set('r@10_test', r_10_test)
                tqdm.write("Val model:{}, epoch:{}, r@1:{:.4f}, r@5:{:.4f}, r@10:{:.4f}".format(self.params.model_name, epoch+1, r_1, r_5, r_10))
                tqdm.write("Test model:{}, epoch:{}, r@1:{:.4f}, r@5:{:.4f}, r@10:{:.4f}".format(self.params.model_name, epoch+1, r_1_test, r_5_test, r_10_test))
                self.logger.log()

                model_to_save = {
                    'model_epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_prev': best_prev
                }
                rr = r_1 + r_5 + r_10
                # save best model
                if rr > best_prev:
                    tqdm.write("Recall at 1 increased....saving best model!!!")
                    best_prev = rr
                    model_to_save['best_prev'] = rr
                    torch.save(model_to_save, os.path.join(self.params.model_path, 'model_w_best.pt'))
                # save model
                if (epoch + 1) % 2 == 0:
                    torch.save(model_to_save, os.path.join(self.params.model_path,'model_w_{}.pt'.format(epoch + 1)))
                self.logger.pop('r@1')
                self.logger.pop('r@5')
                self.logger.pop('r@10')
                self.logger.pop('r@1_test')
                self.logger.pop('r@5_test')
                self.logger.pop('r@10_test')

            # reset batch training loss log
            self.logger.reset('train_loss')
            self.logger.reset('div_loss')
            self.logger.pop('iter')
                       
        
