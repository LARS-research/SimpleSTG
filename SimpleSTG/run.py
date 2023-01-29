
import sys
import random
import os
from os.path import join
import json
import argparse
from datetime import datetime
import pdb
import time

import numpy as np
import pandas as pd
import torch

from data import process_data
from trainer import Trainer
from utils import get_logger, config_gen
from HPspace import *


########## Argparse ##########
args = argparse.ArgumentParser(description='TrafficBench')
args.add_argument('--device', default='cuda:0', type=str)
args.add_argument('--runs', default=1, type=int) # number of samples from HP space
args.add_argument('--dataset', default='pems08', type=str)
args.add_argument('--in_c', default=1, type=int) # 1: raw 2: add time 3: 04/08 3 channels
args.add_argument('--P', default=12, type=int)
args.add_argument('--Q', default=12, type=int)
args.add_argument('--model', default='STGNN', type=str) # STGNN for search

# STGNN specific
args.add_argument('--nblocks', default=6, type=int) 
args.add_argument('--pivot', default="epochs", type=str) # which HP to investigate
args.add_argument('--R', default=None, type=str)  # Specify TS order
args.add_argument('--cfg', default=None, type=str) # Specify all config

args = args.parse_args()
print(args)

def main(config, logger):
  data = process_data(args.dataset, args.in_c, config['batch_size'], P=args.P, Q=args.Q)
  trainer = Trainer(config, args, data, logger)

  try:
    trainer.train()
    mae_um, rmse_um, mae_m, rmse_m, mape_m = trainer.test()
  except:
    print("Error: train/test not finished!")
    mae_um, rmse_um, mae_m, rmse_m, mape_m = torch.Tensor([10000] * 5).cuda()
  return mae_um, rmse_um, mae_m, rmse_m, mape_m


if __name__ == "__main__":
  torch.cuda.set_device(int(args.device[5]))

  HP_ls = eval('rg_'+args.pivot)
  all_res = pd.DataFrame()
  for run in range(args.runs):
    if args.cfg != None:
      with open(args.cfg, 'r') as f:
        config = json.load(f)
    else:
      config = config_gen(args.nblocks)
    if args.R != None:
      config['R_routing'] = args.R 
    
    id_ls, hp_ls, mae_um_ls, rmse_um_ls, mae_m_ls, rmse_m_ls, mape_m_ls = [], [], [], [], [], [], []
    for hp in HP_ls:
      current_time = datetime.now().strftime('%Y%m%d%H%M%S.%f')
      current_dir = os.path.dirname(os.path.realpath(__file__))
      log_dir = join(current_dir,'output', current_time)
      os.makedirs(log_dir)
      args.log_dir = log_dir
      config_dir = join(log_dir, 'config.txt')
      config[args.pivot] = hp

      with open(config_dir, 'w') as f:
        json.dump(config, f, indent=2)
      logger = get_logger(log_dir, name=current_time)

      mae_um, rmse_um, mae_m, rmse_m, mape_m = main(config, logger)

      id_ls.append(str(current_time))
      hp_ls.append(hp)
      mae_um_ls.append(mae_um.cpu())
      rmse_um_ls.append(rmse_um.cpu())
      mae_m_ls.append(mae_m.cpu())
      rmse_m_ls.append(rmse_m.cpu())
      mape_m_ls.append(mape_m.cpu())
      time.sleep(2)

      if mae_um == 10000:
        break

    hp_res = np.stack([mae_um_ls, rmse_um_ls, mae_m_ls, rmse_m_ls, mape_m_ls], axis=1).astype(float)
    hp_res = pd.DataFrame(hp_res, columns=["avg_mae_um","avg_rmse_um","avg_mae_m","avg_rmse_m","avg_mape_m"], index=id_ls)
    hp_res[args.pivot] = hp_ls
    all_res = pd.concat([all_res, hp_res])

  all_res.to_csv('result_'+str(current_time)+'.csv', float_format='%.6f')
  logger.info("\n"+str(all_res))