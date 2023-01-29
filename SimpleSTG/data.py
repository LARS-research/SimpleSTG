import os
import csv
import pickle
import sys
import pdb

import numpy as np
import pandas as pd
import torch

class StandardScaler:
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def transform(self, data):
    return (data - self.mean) / self.std

  def inverse_transform(self, data):
    return (data * self.std) + self.mean

def load_raw(name, in_c):
  """
  wrapper for different datasets
  """
  add_feat = False
  if in_c == 2: add_feat = True

  if name == "metrla":
    df = pd.read_hdf('../../traffic/metrla/metr-la.h5')
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]

    if add_feat == True:
      time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
      time_in_day = np.tile(time_ind, [1, 207, 1]).transpose((2, 1, 0))
      data_list.append(time_in_day)
    data = np.concatenate(data_list, axis=-1)

    pickle_file = '../../traffic/metrla/adj_mx.pkl'
    with open(pickle_file, 'rb') as f:
      sensor_ids, sensor_id_to_ind, adj = pickle.load(f, encoding='latin1')
  elif name == "pemsbay":
    df = pd.read_hdf('../../traffic/pemsbay/pems-bay.h5')
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]

    if add_feat == True:
      time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
      time_in_day = np.tile(time_ind, [1, 325, 1]).transpose((2, 1, 0))
      data_list.append(time_in_day)
    data = np.concatenate(data_list, axis=-1)
    
    pickle_file = '../../traffic/pemsbay/adj_mx_bay.pkl'
    with open(pickle_file, 'rb') as f:
      sensor_ids, sensor_id_to_ind, adj = pickle.load(f, encoding='latin1')
  elif name == "pems04":
    data_path = os.path.join('../../traffic/PEMS04/PEMS04.npz')
    if in_c == 3:
      data = np.load(data_path)['data'][:, :, :3]
    else:
      data = np.load(data_path)['data'][:, :, 0]
      data = np.expand_dims(data, axis=-1)
    data_list = [data]

    if add_feat == True:
      ind = pd.date_range(start='2018-01-01', periods=data.shape[0], freq='5min') 
      time_ind = (ind.values - ind.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
      time_in_day = np.tile(time_ind, [1, data.shape[1], 1]).transpose((2, 1, 0))
      data_list.append(time_in_day)
    data = np.concatenate(data_list, axis=-1)

    N = data.shape[1]
    adj = np.zeros((N, N), dtype=np.float32)
    dist = np.loadtxt('../../traffic/PEMS04/PEMS04.csv', delimiter=',', skiprows=1)
    for _d in dist:
      i, j, d = _d
      adj[int(i), int(j)] = d
    adj[adj != 0] = 1 # without identidy conneciton
  elif name == "pems08":
    data_path = os.path.join('../../traffic/PEMS08/PEMS08.npz')
    if in_c == 3:
      data = np.load(data_path)['data'][:, :, :3]
    else:
      data = np.load(data_path)['data'][:, :, 0]
      data = np.expand_dims(data, axis=-1)
    data_list = [data]

    if add_feat == True:
      ind = pd.date_range(start='2016-07-01', periods=data.shape[0], freq='5min') 
      time_ind = (ind.values - ind.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
      time_in_day = np.tile(time_ind, [1, data.shape[1], 1]).transpose((2, 1, 0))
      data_list.append(time_in_day)
    data = np.concatenate(data_list, axis=-1)

    N = data.shape[1]
    adj = np.zeros((N, N), dtype=np.float32)
    dist = np.loadtxt('../../traffic/PEMS08/PEMS08.csv', delimiter=',', skiprows=1)
    for _d in dist:
      i, j, d = _d
      adj[int(i), int(j)] = d
    adj[adj != 0] = 1 # without identidy conneciton
  elif name == "pems03":
    data_path = os.path.join('../../traffic/PEMS03/PEMS03.npz')
    data = np.load(data_path)['data'][:, :, 0]  # only the first dimension, traffic flow data 
    data = np.expand_dims(data, axis=-1)
    data_list = [data]

    if add_feat == True:
      ind = pd.date_range(start='2018-09-01', periods=data.shape[0], freq='5min') 
      time_ind = (ind.values - ind.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
      time_in_day = np.tile(time_ind, [1, data.shape[1], 1]).transpose((2, 1, 0))
      data_list.append(time_in_day)
    data = np.concatenate(data_list, axis=-1)

    N = data.shape[1]
    adj = np.zeros((N, N), dtype=np.float32)
    dist = np.loadtxt('../../traffic/PEMS03/PEMS03.csv', delimiter=',', skiprows=1)
    id_list = np.loadtxt('../../traffic/PEMS03/PEMS03.txt', dtype='int')
    for _d in dist:
      i, j, d = _d
      adj[int(np.where(id_list==i)[0][0]), int(np.where(id_list==j)[0][0])] = d
    adj[adj != 0] = 1 # without identidy conneciton
  elif name == "pems07":
    data_path = os.path.join('../../traffic/PEMS07/PEMS07.npz')
    data = np.load(data_path)['data'][:, :, 0]  # only the first dimension, traffic flow data 
    data = np.expand_dims(data, axis=-1)
    data_list = [data]

    if add_feat == True:
      ind = pd.date_range(start='2017-05-01', periods=data.shape[0], freq='5min') 
      time_ind = (ind.values - ind.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
      time_in_day = np.tile(time_ind, [1, data.shape[1], 1]).transpose((2, 1, 0))
      data_list.append(time_in_day)
    data = np.concatenate(data_list, axis=-1)

    N = data.shape[1]
    adj = np.zeros((N, N), dtype=np.float32)
    dist = np.loadtxt('../../traffic/PEMS07/PEMS07.csv', delimiter=',', skiprows=1)
    for _d in dist:
      i, j, d = _d
      adj[int(i), int(j)] = d
    adj[adj != 0] = 1 # without identidy conneciton
  elif name == "wind-speed":
    data_path = os.path.join('../../weather/wind-speed/wind-speed.csv')
    data = pd.read_csv(data_path)
    data = np.expand_dims(data, axis=-1)
    
    N = data.shape[1]
    adj = np.ones((N,N))
  elif name == "wind-power":
    data_path = os.path.join('../../weather/wind-power/wind-power.csv')
    data = pd.read_csv(data_path)
    data = np.expand_dims(data, axis=-1)
    
    N = data.shape[1]
    adj = np.ones((N,N))
  elif name == "electricity":
    data_path = os.path.join('../../weather/electricity/electricity.txt')
    data = np.loadtxt(data_path, delimiter=',')
    data = np.expand_dims(data, axis=-1)
    
    N = data.shape[1]
    adj = np.ones((N,N))
  else:
    raise Exception("Check the dataset name!")
  return data, adj # data: (T,N,D), adj: (N,N)

def seq2instance(data, P, Q):
  T, N, C = data.shape
  num_sample = T - P - Q + 1
  x = np.zeros(shape = (num_sample, P, N, C))
  y = np.zeros(shape = (num_sample, Q, N, 1))
  for i in range(num_sample):
    x[i] = data[i : i + P, :, :]
    y[i] = data[i + P : i + P + Q, :, :1]
  x = np.transpose(x, (0, 3, 1, 2))
  y = np.transpose(y, (0, 3, 1, 2))
  return x, y

def loader(X, Y, batch_size=64, shuffle=True):
  X, Y = torch.FloatTensor(X), torch.FloatTensor(Y)
  data = torch.utils.data.TensorDataset(X, Y)
  dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
  return dataloader

def process_data(name, in_c=1, batch_size=64, tr_ratio=0.7, te_ratio=0.2, P=12, Q=12):
  if name in ['pems03', 'pems04', 'pems07', 'pems08']:
    tr_ratio = 0.6
    # add_feat = True
  elif name in ['metrla', 'pemsbay']:
    tr_ratio = 0.7
    # add_feat = True
  elif name in ['wind-speed', 'wind-power', 'electricity']:
    tr_ratio = 0.7
  else:
    raise Exception("Wrong dataset name.")

  data, adj = load_raw(name, in_c) # data: T,N,C
  print(data.shape)

  T = data.shape[0]
  train_steps = round(tr_ratio * T)
  test_steps = round(te_ratio * T)
  val_steps = T - train_steps - test_steps

  # all train/val/test should have shape: T,N,C
  train = data[: train_steps]
  val = data[train_steps : train_steps + val_steps]
  test = data[-test_steps :]

  scaler = StandardScaler(mean=train[:,:,0].mean(), std=train[:,:,0].std())

  data = {}
  data['x_train'], data['y_train'] = seq2instance(train, P, Q)
  data['x_val'], data['y_val'] = seq2instance(val, P, Q)
  data['x_test'], data['y_test'] = seq2instance(test, P, Q)

  # all train/val/test should have shape: B,C,P/Q,N
  print('Train:\t', data['x_train'].shape, data['y_train'].shape)
  print('Val:\t', data['x_val'].shape, data['y_val'].shape)
  print('Test:\t', data['x_test'].shape, data['y_test'].shape)

  # Normalization
  for category in ['train', 'val', 'test']:
    data['x_' + category][:,0,:,:] = scaler.transform(data['x_' + category][:,0,:,:])
    data['y_' + category][:,0,:,:] = scaler.transform(data['y_' + category][:,0,:,:]) # uncomment

  # DataLoader
  dl = {}
  dl['adj'] = adj
  dl['train_loader'] = loader(data['x_train'], data['y_train'], batch_size=batch_size, shuffle=True)
  dl['val_loader'] = loader(data['x_val'], data['y_val'], batch_size=batch_size, shuffle=False)
  dl['test_loader'] = loader(data['x_test'], data['y_test'], batch_size=batch_size, shuffle=False)
  dl['scaler'] = scaler
  
  return dl

