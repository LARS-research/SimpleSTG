import os
import time
import math
import sys
import pdb
import json

import numpy as np
import torch 

from model import STGNN
from utils import count_model_parameters, mae_loss, masked_mae_loss, all_metrics

class Trainer(object):

  def __init__(self, config, args, data, logger):
    
    self.config = config
    self.args = args
    self.logger = logger

    self.device = args.device
    self.mask_zero = True if args.dataset in ['metrla', 'pemsbay'] else False
    self.best_dir = os.path.join(args.log_dir, 'best_model.pth')

    # data
    self.dl_train, self.dl_val, self.dl_test, self.scaler, self.adj = data['train_loader'], data['val_loader'], data['test_loader'], data['scaler'], data['adj']

    # curriculum learning
    self.out_dim = args.Q
    self.cl_steps = config['cl_steps']
    # self.cl_steps = None
    if self.cl_steps != None:
      self.task_level = 1
    else:
      self.task_level = self.out_dim

    # model
    for batch_idx, (data, target) in enumerate(self.dl_train): 
      self.in_c = data.shape[1]
      break
    model = self._model_init(args.model)
    self.model = model.to(args.device)

    if config['optim'] == 'sgd':
      self.optim = torch.optim.SGD(params=self.model.parameters(), lr=config['lr'], weight_decay=config['wdecay'])
    elif config['optim'] == 'rmsprop':
      self.optim = torch.optim.RMSprop(params=self.model.parameters(), lr=config['lr'], weight_decay=config['wdecay'])
    elif config['optim'] == 'adam':
      self.optim = torch.optim.Adam(params=self.model.parameters(), lr=config['lr'], weight_decay=config['wdecay'])
    elif config['optim'] == 'adamw':
      self.optim = torch.optim.AdamW(params=self.model.parameters(), lr=config['lr'], weight_decay=config['wdecay'])
    elif config['optim'] == 'adamax':
      self.optim = torch.optim.Adamax(params=self.model.parameters(), lr=config['lr'], weight_decay=config['wdecay'])
    else:
      raise Exception("Optim name wrong!")

    total_num_param = count_model_parameters(self.model, True)
    self.logger.info('Total params num: {}'.format(total_num_param))
  
  def _model_init(self, name):
    if name == 'STGNN':
      model = STGNN(N=self.adj.shape[0], in_c=self.in_c, config=self.config, adj=self.adj)
    else:
      raise Exception("Model name wrong, check args.model!")
    return model

  def train(self):
    best_loss = float('inf')
    not_improved_count = 0
    start_time = time.time()
    for ep in range(1, self.config['epochs']+1):
      train_epoch_loss, val_epoch_loss = self.train_epoch(ep) #TODO
      if val_epoch_loss < best_loss:
        best_ep = ep
        best_loss = val_epoch_loss
        not_improved_count = 0
        self.save_checkpoint()
        self.test()
      else:
        not_improved_count += 1

      if (train_epoch_loss > 100) & (ep >= 10):
        self.logger.info("Detected divergence! Stop at epoch 10!")
        break
    
    training_time = time.time() - start_time
    self.logger.info("Total training time: {:.1f} mins".format(training_time/60))

  def train_epoch(self, epoch):
    self.model = self.model.train()
    total_train_loss = 0
    if self.cl_steps != None:
      if epoch%self.cl_steps==0 and self.task_level<self.out_dim:
          self.task_level += 1

    for batch_idx, (data, target) in enumerate(self.dl_train): # data, target, BCTN, B1TN
      label = target[:,0,:,:].squeeze().to(self.device) # label shape: BTN
      label = torch.transpose(label, 0, 1) # label shape: TBN
      #fix label
      if len(label.shape) == 2:
        label = label.permute(1,0).unsqueeze(1)
      data = data.to(self.device) # input shape: BCTN
      output = self.model(data) # output shape: TODO
            
      self.optim.zero_grad()
      loss = self._compute_loss_mse( output[:self.task_level,...], label[:self.task_level,...], self.mask_zero)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_norm'])
      self.optim.step()

      total_train_loss += loss.item()

    train_epoch_loss = total_train_loss / len(self.dl_train)
    val_epoch_loss = self.val_epoch(self.dl_val)
    test_epoch_loss = self.val_epoch(self.dl_test)
    self.logger.info('***** Epoch {}, Train loss:{:.3f}, Val loss:{:.3f}, Test loss:{:.3f}'.format(epoch, train_epoch_loss, val_epoch_loss, test_epoch_loss))
    return train_epoch_loss, val_epoch_loss
  
  def val_epoch(self, dl):
    self.model = self.model.eval()
    total_val_loss = 0
    with torch.no_grad():
      for batch_idx, (data, target) in enumerate(dl):
        label = target[:,0,:,:].squeeze().to(self.device)
        label = torch.transpose(label, 0, 1) 
        
        #fix label
        if len(label.shape) == 2:
          label = label.permute(1,0).unsqueeze(1)
          
        data = data.to(self.device)
        output = self.model(data)
        
        loss = self._compute_loss_mse(output, label, self.mask_zero)
        total_val_loss += loss.item()
    val_loss = total_val_loss / len(dl)

    return val_loss 
  
  def test(self):
    best_model = self._model_init(self.args.model)
    best_model.load_state_dict(torch.load(self.best_dir))
    best_model = best_model.to(self.device)
    best_model = best_model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
      for batch_idx, (data, target) in enumerate(self.dl_test):
        label = target[:,0,:,:].squeeze().to(self.device) 
        label = torch.transpose(label, 0, 1)
        data = data.to(self.device) # input shape: B, C, T, N
        output = best_model(data)
        
        #fix label
        if len(label.shape) == 2:
          label = label.permute(1,0).unsqueeze(1)
          
        y_true.append(label)
        y_pred.append(output)
  
    y_pred = torch.cat(y_pred, dim=1)
    y_true = torch.cat(y_true, dim=1)

    for t in range(y_pred.shape[0]):
      metrics = self._compute_metrics(y_pred[t], y_true[t])
      if (t+1) in [3,6,12]:
        log = 'Test horizon {:2d}, Unmasked MAE:{:.4f}, Unmasked RMSE:{:.4f}, Masked MAE:{:.4f}, Masked RMSE:{:.4f}, Masked MAPE:{:.4f}'
        self.logger.info(log.format(t+1, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]*100))
    mae_um, rmse_um, mae_m, rmse_m, mape_m = self._compute_metrics(y_pred, y_true)
    self.logger.info("Avg Horizon, Unmasked MAE:{:.4f}, Unmasked RMSE:{:.4f}, Masked MAE:{:.4f}, Masked RMSE:{:.4f}, Masked MAPE:{:.4f}".format(mae_um, rmse_um, mae_m, rmse_m, mape_m*100))
    return mae_um, rmse_um, mae_m, rmse_m, mape_m 

  def _compute_loss_mse(self, y_pred, y_true, mask_zero):
    y_true = self.scaler.inverse_transform(y_true)
    y_pred = self.scaler.inverse_transform(y_pred)

    if mask_zero:
      return masked_mae_loss(y_pred, y_true)
    else:
      return mae_loss(y_pred, y_true)
  
  def _compute_metrics(self, y_pred, y_true):
    y_true = self.scaler.inverse_transform(y_true)
    y_pred = self.scaler.inverse_transform(y_pred)
    return all_metrics(y_pred, y_true)

  def save_checkpoint(self):
    torch.save(self.model.state_dict(), self.best_dir)
