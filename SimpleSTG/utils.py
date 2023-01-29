
from os.path import join
import logging
from datetime import datetime
import random
from random import choice as rc
import json
import sys
import re

import numpy as np
import torch

from HPspace import *


########## General ##########
def seed_init(seed): 
  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.cudnn_enabled = False
  torch.backends.cudnn.deterministic = True
  

def count_model_parameters(model, only_num = True):
  if not only_num:
    for name, param in model.named_parameters():
      print("{:30}, {}, {}, \t{}".format(str(param.shape), param.requires_grad, param.nelement(), name))
  total_num = sum([param.nelement() for param in model.parameters()])
  return total_num


########## Logging ##########
def get_logger(log_path, name=None):
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)

  # Define the format
  formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d}: %(message)s', "%Y-%m-%d %H:%M")
  
  # Create handler for logging to console
  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.DEBUG)
  console_handler.setFormatter(formatter)

  # Create handler for logging to file
  logfile = join(log_path, 'run.log')
  print('Create Log File in: ', logfile)
  file_handler = logging.FileHandler(logfile, mode='w')
  file_handler.setLevel(logging.DEBUG)
  file_handler.setFormatter(formatter)
  
  # Add Handler to logger
  logger.addHandler(console_handler)
  logger.addHandler(file_handler)

  return logger


########## Losses ##########
def mae_loss(y_pred, y_true):
  # print("y_true:", y_true.shape, "y_pred:", y_pred.shape)
  assert y_pred.shape == y_true.shape
  return torch.nn.L1Loss()(y_pred, y_true).mean()

def mse_loss(y_pred, y_true):
  assert y_pred.shape == y_true.shape
  return torch.nn.MSELoss()(y_pred, y_true).mean()

def masked_mae_loss(y_pred, y_true):
  # modified from https://github.com/nnzhan/MTGNN/blob/master/util.py)
  assert y_pred.shape == y_true.shape
  mask = (y_true != 0).float()
  mask /= torch.mean(mask)
  loss = torch.abs(y_pred - y_true)
  loss = loss * mask
  # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
  loss[loss != loss] = 0
  return loss.mean()

########## Metrics ##########
def rmse(preds, labels):
  assert preds.shape == labels.shape 
  return torch.sqrt(torch.nn.MSELoss()(preds, labels).mean())

# Reference: https://github.com/nnzhan/MTGNN/blob/master/util.py
def masked_rmse(preds, labels, null_val=0.0):
  if np.isnan(null_val):
    mask = ~torch.isnan(labels)
  else:
    mask = (labels!=null_val)
  mask = mask.float()
  mask /= torch.mean((mask))
  mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
  loss = (preds-labels)**2
  loss = loss * mask
  loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
  return torch.sqrt(torch.mean(loss))

def masked_mae(preds, labels, null_val=0.0):
  if np.isnan(null_val):
    mask = ~torch.isnan(labels)
  else:
    mask = (labels!=null_val)
  mask = mask.float()
  mask /=  torch.mean((mask))
  mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
  loss = torch.abs(preds-labels)
  loss = loss * mask
  loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
  return torch.mean(loss)

# Reference: https://github.com/LeiBAI/AGCRN/blob/master/lib/metrics.py
def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def MSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))
  
def MAPE_torch(pred, true, mask_value=0.0):
  mask = torch.gt(true, mask_value)
  pred = torch.masked_select(pred, mask)
  true = torch.masked_select(true, mask)    
  return torch.mean(torch.abs(torch.div((true - pred), true)))

def all_metrics(preds, labels):
  # mae_um =  mae_loss(preds, labels)
  # rmse_um = rmse(preds, labels)
  
  mae_um = MAE_torch(preds, labels, 0.0)
  rmse_um = RMSE_torch(preds, labels, 0.0)
  mae_m = masked_mae(preds, labels)
  rmse_m = masked_rmse(preds, labels)
  mape_m = MAPE_torch(preds, labels)
  return mae_um, rmse_um, mae_m, rmse_m, mape_m


########## Config generation ##########
def config_gen(num_blocks=6): # each block is T or S
  config = {}
  hp_general = {
    "epochs": rc(rg_epochs),
    "early_stop": rc(rg_early_stop),
    "lr": rc(rg_lr),
    "lr_decay_steps": rc(rg_lr_decay_steps),
    "lr_decay_ratio": rc(rg_lr_decay_ratio),
    "batch_size": rc(rg_batch_size),
    "grad_norm": rc(rg_grad_norm),
    "optim": rc(rg_optim), 
    "wdecay": rc(rg_wdecay),
    "loss": rc(rg_loss),
    "dropout": rc(rg_dropout),
    "mid_channel": rc(rg_mid_channel),
    "skip_channel": rc(rg_skip_channel),
    "end_channel": rc(rg_end_channel),
    "cl_steps": rc(rg_cl_steps),

    "G_method": rc(rg_G_method),
    "G_k": rc(rg_G_k),
    "G_dim": rc(rg_G_dim),
    "G_alpha": rc(rg_G_alpha),
    "G_mix": rc(rg_G_mix),
  }
  
  ks_len = rc(rg_T_ks_len)
  ks = sorted([rc(rg_T_ks) for _ in range(ks_len)])

  hp_st = {
    "S_hop": rc(rg_S_hop),
    "S_fusion": rc(rg_S_fusion),
    "S_hopalpha": rc(rg_S_hopalpha),

    "T_dilation": rc(rg_T_dilation),
    "T_ks": ks,

    "R_skip": rc(rg_R_skip),
    "R_residual": rc(rg_R_residual),
  }

  routing = ''.join(['T' if s == '1' else 'S' for s in "{:06b}".format(np.random.randint(0, 2**6))])
  skip_enc = "{:021b}".format(np.random.randint(0, 2**21))

  config = {**hp_general, **hp_st}
  config['R_routing'] = routing
  config['R_skip_enc'] = skip_enc

  return config