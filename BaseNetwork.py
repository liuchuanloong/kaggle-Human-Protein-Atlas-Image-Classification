import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F

from contextlib import contextmanager
import datetime
import  time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Scheduler
from Analysis import show_image_mask, show_image_mask_pred, show_image_tta_pred
from Evaluation import  do_acc, F1_score_batch, best_threshold_score1, best_threshold_score2
from Loss import FocalLoss1, FocalLoss2
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.3f}s".format(title, time.time() - t0))

class BaseNetwork(nn.Module):

    def __init__(self, lr=0.005, fold=None, val_mode='max', comment=''):
        super(BaseNetwork, self).__init__()
        self.lr = lr
        self.fold = fold
        self.scheduler = None
        self.best_model_path = None
        self.epoch = 0
        self.val_mode = val_mode
        self.comment = comment

        if self.val_mode == 'max':
            self.best_metric = -np.inf
        elif self.val_mode == 'min':
            self.best_metric = np.inf

        self.train_log = dict(loss=[], acc=[], f1_score=[])
        self.val_log = dict(loss=[], acc=[], f1_score=[], best_f1_score=[])
        self.create_save_folder()

    def create_optmizer(self, optimizer='SGD', use_scheduler=None, gamma=0.25, patience=4,
                        milestones=None, T_max=10, T_mul=2, lr_min=0):
        self.cuda()
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                  lr=self.lr, momentum=0.9, weight_decay=0.0001)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                       self.parameters()), lr=self.lr)

        if use_scheduler == 'ReduceOnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                  mode='max',
                                                                  factor=gamma,
                                                                  patience=patience,
                                                                  verbose=True,
                                                                  threshold=0.01,
                                                                  min_lr=1e-05,
                                                                  eps=1e-08)

        elif use_scheduler == 'Milestones':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=milestones,
                                                            gamma=gamma,
                                                            last_epoch=-1)

        elif use_scheduler == 'CosineAnneling':
            self.scheduler = Scheduler.CosineAnnealingLR(self.optimizer,
                                                         T_max=T_max,
                                                         T_mul=T_mul,
                                                         lr_min=lr_min,
                                                         val_mode=self.val_mode,
                                                         last_epoch=-1,
                                                         save_snapshots=True)


    def train_network(self, train_loader, val_loader, n_epoch=10, earlystopping_patience=None):
        print('Model created, total of {} parameters'.format(
            sum(p.numel() for p in self.parameters())))
        self.best_score = float('inf')
        self.earlystopping_patience = earlystopping_patience
        while self.epoch < n_epoch:
            self.epoch += 1
            lr = np.mean([param_group['lr'] for param_group in self.optimizer.param_groups])
            with timer('Train Epoch {:}/{:} - LR: {:.3E}'.format(self.epoch, n_epoch, lr)):
                # Training step
                train_loss, train_acc, train_f1_score = self.training_step(train_loader)
                #  Validation
                val_loss, val_acc, val_f1_score, best_f1_score, best_th = self.perform_validation(val_loader)
                # Learning Rate Scheduler
                if self.scheduler is not None:
                    if type(self.scheduler).__name__ == 'ReduceLROnPlateau':
                        self.scheduler.step(np.mean(val_loss))
                    elif type(self.scheduler).__name__ == 'CosineAnnealingLR':
                        self.scheduler.step(self.epoch,
                                            save_dict=dict(metric=np.mean(val_loss),
                                                           save_dir=self.save_dir,
                                                           best_th = best_th,
                                                           fold=self.fold,
                                                           state_dict=self.state_dict()))
                    else:
                        self.scheduler.step(self.epoch)
                # Save best model
                if type(self.scheduler).__name__ != 'CosineAnnealingLR':
                    save_model = dict(best_th = best_th,
                                      state_dict = self.state_dict)
                    self.save_best_model(np.mean(val_loss), save_model)

            # Print statistics
            print(('train loss: {:.3f}  val_loss: {:.3f}  '
                   'train acc:  {:.3f}  val_acc:  {:.3f}  '
                   'train f1_score:  {:.3f}  val_f1_score:  {:.3f}  '
                   'val_f1_score(best threshold): {:.3f}').format(
                np.mean(train_loss),
                np.mean(val_loss),
                np.mean(train_acc),
                np.mean(val_acc),
                np.mean(train_f1_score),
                np.mean(val_f1_score),
                np.mean(best_f1_score)))
            # Early Stopping Scheduler
            if self.earlystopping_patience is not None:
                score = np.mean(val_loss)
                if score > self.best_score:
                    self.counter += 1
                    print("EarlyStopping: %i / %i" % (self.counter, self.earlystopping_patience))
                    if self.counter >= self.earlystopping_patience:
                        print("EarlyStopping: Stop training")
                        break
                else:
                    print('Best score improve %f to %f' % (self.best_score, score))
                    self.best_score = score
                    self.counter = 0

        self.save_training_log()

    def training_step(self, train_loader):
        self.set_mode('train')
        train_loss = []
        train_acc = []
        train_f1_score = []
        for i, data in enumerate(train_loader):

            self.optimizer.zero_grad()
            imgs = data[0].cuda()
            labels = data[1].cuda()
            logit = self.forward(imgs)

            # logit = logit.cpu()
            # labels = labels.cpu()

            pred = torch.sigmoid(logit)
            # img_out = torch.sigmoid(img_out)
            loss = self.criterion(logit, labels)
            # loss2 = F.mse_loss(img_out, imgs)
            # loss = 0.1*loss1 + loss2

            acc  = do_acc(pred, labels, th=0.5)
            f1_score = F1_score_batch(pred.cpu().data.numpy(), labels.cpu().data.numpy(), threshold=0.5)

            train_loss.append(loss.item())
            train_acc.append(acc)
            train_f1_score.append(f1_score)

            loss.backward()
            self.optimizer.step()

        # Append epoch data to metrics dict
        for metric in ['loss', 'acc', 'f1_score']:
            self.train_log[metric].append(np.mean(eval('train_{}'.format(metric))))
        return train_loss, train_acc, train_f1_score


    def perform_validation(self, val_loader):
        self.set_mode('valid')
        val_loss = []
        val_acc = []
        val_f1_score = []
        val_preds = []
        val_trues = []
        for imgs, labels in val_loader:
            imgs = imgs.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                # logit, img_out = self.forward(imgs)
                logit = self.forward(imgs)

                # logit = logit.cpu()
                # labels = labels.cpu()

                pred = torch.sigmoid(logit)
                # img_out = torch.sigmoid(img_out)

                loss = self.criterion(logit, labels)
                # loss2 = F.mse_loss(img_out, imgs)
                # loss = 0.1*loss1 + loss2
                acc  = do_acc(pred, labels, th=0.5)
                f1_score = F1_score_batch(pred.cpu().data.numpy(), labels.cpu().data.numpy(), threshold=0.5)

            val_loss.append(loss.item())
            val_acc.append(acc)
            val_f1_score.append(f1_score)
            val_preds.extend(pred.cpu().data.numpy())
            val_trues.extend(labels.cpu().data.numpy())
        val_best_f1_score, best_th = best_threshold_score2(np.array(val_preds), np.array(val_trues))
        print('Treshold: %s'%best_th)
        # Append epoch data to metrics dict
        for metric in ['loss', 'acc', 'f1_score', 'best_f1_score']:
            self.val_log[metric].append(np.mean(eval('val_{}'.format(metric))))

        return val_loss, val_acc, val_f1_score, val_best_f1_score, best_th

    def predict(self, test_loader, threshold=0.5):
        self.set_mode('test')
        self.cuda()
        debug = False
        preds = []
        for i, imgs in enumerate(test_loader):
            with torch.no_grad():
                # Predict original batch
                imgs = imgs[0].cuda()
                logit = self.forward(imgs)
                pred = torch.sigmoid(logit)
                # img_out = torch.sigmoid(img_out)
                # if debug:
                #     fig, axis = plt.subplots(2,4, figsize=(20,20))
                #     axis[0,0].imshow(img_out[0,0,:,:])
                #     axis[0,1].imshow(img_out[0,1,:,:])
                #     axis[0,2].imshow(img_out[0,2,:,:])
                #     axis[0,3].imshow(img_out[0,3,:,:])
                #     axis[1, 0].imshow(imgs[0, 0, :, :])
                #     axis[1, 1].imshow(imgs[0, 1, :, :])
                #     axis[1, 2].imshow(imgs[0, 2, :, :])
                #     axis[1, 3].imshow(imgs[0, 3, :, :])
                #     plt.show()
                pred = pred.cpu().data.numpy() > threshold
                preds.extend(pred)
        return preds

    def define_criterion(self, name, weights=None):
        if name.lower() == 'crossentropyloss':
            if weights is not None:
                self.criterion = nn.CrossEntropyLoss(pos_weight=weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif name.lower() == 'focalloss1':
            self.criterion = FocalLoss1()
        elif name.lower() == 'focalloss2':
            self.criterion = FocalLoss2()
        elif name.lower() == 'bceloss':
            if weights is not None:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('Loss {} is not implemented'.format(name))


    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


    def save_best_model(self, metric, save_model):
        if (self.val_mode == 'max' and metric > self.best_metric) or (self.val_mode == 'min' and metric < self.best_metric):
            # Update best metric
            self.best_metric = metric
            # Remove old file
            if self.best_model_path is not None:
                os.remove(self.best_model_path)
            # Save new best model weights
            date = ':'.join(str(datetime.datetime.now()).split(':')[:2])
            if self.fold is not None:
                self.best_model_path = os.path.join(
                    self.save_dir,
                    '{:}_Fold{:}_Epoach{}_val{:.3f}'.format(date, self.fold, self.epoch, metric))
            else:
                self.best_model_path = os.path.join(
                    self.save_dir,
                    '{:}_Epoach{}_val{:.3f}'.format(date, self.epoch, metric))

            torch.save(save_model, self.best_model_path)


    def save_training_log(self):
        d = dict()
        for tk, vk in zip(self.train_log.keys(), self.val_log.keys()):
            d['train_{}'.format(tk)] = self.train_log[tk]
            d['val_{}'.format(vk)] = self.val_log[vk]

        df = pd.DataFrame(d)
        df.index += 1
        df.index.name = 'Epoach'

        date = ':'.join(str(datetime.datetime.now()).split(':')[:2])
        if self.fold is not None:
            p = os.path.join(
                self.save_dir,
                '{:}_Fold{:}_TrainLog.csv'.format(date, self.fold))
        else:
            p = os.path.join(
                self.save_dir,
                '{:}_TrainLog.csv'.format(date))

        df.to_csv(p, sep=";")

        with open(p, 'a') as fd:
            fd.write(self.comment)


    def load_model(self, path=None, best_model=False):
        if best_model:
            save_model = torch.load(self.best_model_path)
            self.load_state_dict(save_model['state_dict'], strict=False)
            return save_model['best_th']
        else:
            save_model = torch.load(path)
            self.load_state_dict(save_model['state_dict'], strict=False)
            return save_model['best_th']

    def create_save_folder(self):
        name = type(self).__name__
        self.save_dir = os.path.join('./Saves', name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def plot_training_curve(self, show=True):
        fig, axs = plt.subplots(3, 1)
        for i, metric in enumerate(['loss', 'acc', 'f1_score']):
            axs[i].plot(self.train_log[metric], 'r', label='Train')
            axs[i].plot(self.val_log[metric], 'b', label='Validation')
            if metric is 'f1_score':
                axs[i].plot(self.val_log['best_f1_score'], 'y', label='val(best threshold)')
                max = np.argmax(self.val_log['best_f1_score'])
                axs[i].plot(max, self.val_log['best_f1_score'][max], "xg", label='best_{}'.format(metric))
            if metric is 'loss':
                min = np.argmin(self.val_log[metric])
                axs[i].plot(min, self.val_log[metric][min], "xg", label='best_loss')
            if metric is 'acc':
                max = np.argmax(self.val_log[metric])
                axs[i].plot(max, self.val_log[metric][max], "xg", label='best_{}'.format(metric))

            axs[i].legend()
            axs[i].set_title(metric)
            axs[i].set_xlabel('Epochs')
            axs[i].set_ylabel(metric)
        if show:
            plt.show()
