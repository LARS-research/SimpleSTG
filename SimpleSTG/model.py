import pdb
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# Basic Layer
################################################################################

class linear(nn.Module):
  def __init__(self,c_in,c_out):
    super(linear,self).__init__()
    self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

  def forward(self,x):
    return self.mlp(x)

class nconv(nn.Module):
  def __init__(self):
    super(nconv,self).__init__()

  def forward(self, x, A):
    x = torch.einsum('ncwl,vw->ncvl',(x,A))
    return x.contiguous()

################################################################################
# S Layer
################################################################################

class MixHop(nn.Module):
  def __init__(self,c_in,c_out,hop,hopalpha,fusion):
    super(MixHop, self).__init__()
    self.nconv = nconv()
    self.hop = hop
    self.alpha = hopalpha
    self.fusion = fusion

    # self.mlp = linear((hop+1)*c_in,c_out)
    if fusion == 'concat':
      self.out = linear((hop+1)*c_in,c_out)
    elif fusion in ['sum', 'mean', 'max']:
      self.out = linear(c_in,c_out)
    self.mlp = nn.ModuleList()
    for i in range(hop):
      self.mlp.append(linear(c_in,c_out))

  def forward(self,x,adj):
    adj = adj + torch.eye(adj.shape[0]).to(x.device) # TODO: how to preprocess adj
    d = adj.sum(1)
    a = adj / d.view(-1, 1)

    h = x
    out = [h]
    for i in range(self.hop):
      h = self.alpha*x + (1-self.alpha)*self.nconv(h,a) # TODO: x?
      out.append(self.mlp[i](h))
    if self.fusion == 'concat':
      ho = self.out(torch.cat(out,dim=1))
    elif self.fusion == 'sum':
      ho = self.out(torch.sum(out))
    else:
      raise Exception('S_fusion wrong!')
    return ho

class S_MixHop(nn.Module):
  def __init__(self,cin,cout,hop,hopalpha,fusion,G_mix):
    super(S_MixHop, self).__init__()
    self.gconvf1 = MixHop(cin,cout,hop,hopalpha,fusion)
    self.gconvf2 = MixHop(cin,cout,hop,hopalpha,fusion)
    self.gconvb1 = MixHop(cin,cout,hop,hopalpha,fusion)
    self.gconvb2 = MixHop(cin,cout,hop,hopalpha,fusion)
    self.G_mix = G_mix

  def forward(self, x, adj): # BCNT -> BCNT
    assert len(adj) == 2 # adj[0]:GT, adj[1]: param
    x0 = self.gconvf1(x, adj[0])+self.gconvb1(x, adj[0].transpose(1,0)) # GT adj
    x1 = self.gconvf2(x, adj[1])+self.gconvb2(x, adj[1].transpose(1,0)) # param adj

    x = self.G_mix * x1 + (1-self.G_mix) * x0 # G_mix 0: only GT
    return x 

################################################################################
# T Layer
################################################################################

class T_Inception(nn.Module):
  def __init__(self,cin,cout,kernel_set,dilation_factor,dropout):
    super(T_Inception, self).__init__()
    assert cout % len(kernel_set) == 0
    self.kernel_set = kernel_set # [2,3,6,7]
    self.tconv_f = nn.ModuleList()
    self.tconv_g = nn.ModuleList()
    cout = int(cout/len(self.kernel_set))
    for kern in self.kernel_set:
      self.tconv_f.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))
      self.tconv_g.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))
    self.do = nn.Dropout(dropout)

  def forward(self,input):
    f, g = [], []
    for i in range(len(self.kernel_set)):
      f.append(self.tconv_f[i](input))
      g.append(self.tconv_g[i](input))
    for i in range(len(self.kernel_set)):
      f[i] = f[i][...,-f[-1].size(3):]
      g[i] = g[i][...,-g[-1].size(3):]
    f = torch.cat(f,dim=1)
    g = torch.cat(g,dim=1)

    out = self.do(torch.tanh(f)*torch.sigmoid(g))
    return out

################################################################################
# G Layer
################################################################################

class G_MTGNN(nn.Module):
  def __init__(self, N, k, dim, alpha=3):
    super(G_MTGNN, self).__init__()
    self.N = N
    self.emb1 = nn.Embedding(N, dim)
    self.emb2 = nn.Embedding(N, dim)
    self.lin1 = nn.Linear(dim,dim)
    self.lin2 = nn.Linear(dim,dim)

    self.k = k # kNN
    self.dim = dim # Embed dim
    self.alpha = alpha

  def forward(self, idx):
    nodevec1 = self.emb1(idx)
    nodevec2 = self.emb2(idx)

    nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
    nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

    a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
    adj = F.relu(torch.tanh(self.alpha*a))
    mask = torch.zeros(idx.size(0), idx.size(0)).cuda()
    mask.fill_(float('0'))
    s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
    mask.scatter_(1,t1,s1.fill_(1))    
    adj = adj*mask
    return adj

################################################################################
# STGR Compilation
################################################################################

class STGNN(nn.Module):
  def __init__(self, N, in_c, config, adj, in_dim=12, out_dim=12):
    super(STGNN, self).__init__()

    self.N = N
    self.adj = torch.Tensor(adj).cuda()

    if config['G_method'] == 'mtgnn':
      self.gc = G_MTGNN(self.N, config['G_k'], config['G_dim'], config['G_alpha'])
      self.idx = torch.arange(self.N).cuda()
    else:
      raise Exception("Wrong G_method!")
    self.mid_c = config['mid_channel']
    self.skip_c = config['skip_channel']
    self.end_c = config['end_channel']
    self.dropout = config['dropout']
    self.do = nn.Dropout(self.dropout)
    self.routing = list(config['R_routing']) # e.g. ['T', 'S', 'T']
    self.start_conv = linear(in_c,self.mid_c)
    self.end_conv_1 = linear(self.skip_c,self.end_c) # Skip connection
    self.end_conv_2 = linear(self.end_c,out_dim)

    self.R_skip = config['R_skip']
    self.R_residual = config['R_residual']
    self.R_skip_enc = config['R_skip_enc']

    self.n_blocks = len(self.routing)
    self.n_T = self.routing.count('T')
    max_ks = max(config['T_ks'])
    dilation = config['T_dilation']
    self.receptive_field = 1 + self.n_T * (max_ks-1)*dilation

    self.blocks = nn.ModuleList() # each block is either T or S
    self.norm = nn.ModuleList()
    self.skipT = nn.ModuleList()
    self.skipS = nn.ModuleList() 
    rf_curr = 1 
    for i, blk in enumerate(self.routing):
      if blk == 'T':
        rf_curr += (max_ks-1)*dilation
        self.blocks.append(T_Inception(self.mid_c,self.mid_c,config['T_ks'],dilation,self.dropout))
        self.skipT.append(nn.Conv2d(self.mid_c, self.skip_c, kernel_size=(1, max(in_dim,self.receptive_field)-rf_curr+1)))
      elif blk == 'S':
        self.blocks.append(S_MixHop(self.mid_c,self.mid_c,config['S_hop'],config['S_hopalpha'],config['S_fusion'],config['G_mix'])) 
        self.skipS.append(nn.Conv2d(self.mid_c,self.skip_c,kernel_size=(1, max(in_dim,self.receptive_field)-rf_curr+1)))
        self.norm.append(nn.LayerNorm((self.mid_c, self.N, max(in_dim,self.receptive_field)-rf_curr+1)))
      else:
        raise Exception("Wrong block!") 
    
    self.skipInput = nn.Conv2d(in_c, self.skip_c, kernel_size=(1,max(in_dim,self.receptive_field)))

  def forward(self, input): # Model: BCTN -> TBN
    input = input.permute(0,1,3,2) # BCTN -> BCNT
    seq_len = input.size(3) # T
    if seq_len<self.receptive_field:
      input = F.pad(input,(self.receptive_field-seq_len,0,0,0))
    adj = [self.adj, self.gc(self.idx)]

    skip_Input = self.skipInput(self.do(input))
    x = self.start_conv(input)
    skip_T, skip_S = 0, 0
    i_t, i_s = 0, 0

    res_list = []
    skip_list = []
    for i, blk in enumerate(self.routing):
      res_list.append(x)

      if blk == 'T':
        x = self.blocks[i](x)
        skip_list.append(self.skipT[i_t](x))
        i_t += 1
      elif blk == 'S':
        x = self.blocks[i](x, adj)
        skip_list.append(self.skipS[i_s](x))
        x = self.norm[i_s](x)
        i_s += 1 

      if i == 0:
        if self.R_skip_enc[0] == '1': x = x + res_list[0][:, :, :, -x.size(3):]
      elif i == 1:
        if self.R_skip_enc[1] == '1': x = x + res_list[0][:, :, :, -x.size(3):]
        if self.R_skip_enc[2] == '1': x = x + res_list[1][:, :, :, -x.size(3):]
      elif i == 2:
        if self.R_skip_enc[3] == '1': x = x + res_list[0][:, :, :, -x.size(3):]
        if self.R_skip_enc[4] == '1': x = x + res_list[1][:, :, :, -x.size(3):]
        if self.R_skip_enc[5] == '1': x = x + res_list[2][:, :, :, -x.size(3):]
      elif i == 3:
        if self.R_skip_enc[6] == '1': x = x + res_list[0][:, :, :, -x.size(3):]
        if self.R_skip_enc[7] == '1': x = x + res_list[1][:, :, :, -x.size(3):]
        if self.R_skip_enc[8] == '1': x = x + res_list[2][:, :, :, -x.size(3):]
        if self.R_skip_enc[9] == '1': x = x + res_list[3][:, :, :, -x.size(3):]
      elif i == 4:
        if self.R_skip_enc[10] == '1': x = x + res_list[0][:, :, :, -x.size(3):]
        if self.R_skip_enc[11] == '1': x = x + res_list[1][:, :, :, -x.size(3):]
        if self.R_skip_enc[12] == '1': x = x + res_list[2][:, :, :, -x.size(3):]
        if self.R_skip_enc[13] == '1': x = x + res_list[3][:, :, :, -x.size(3):]
        if self.R_skip_enc[14] == '1': x = x + res_list[4][:, :, :, -x.size(3):]
      elif i == 5:
        skip = 0
        if self.R_skip_enc[15] == '1': skip += skip_Input
        if self.R_skip_enc[16] == '1': skip += skip_list[0]
        if self.R_skip_enc[17] == '1': skip += skip_list[1]
        if self.R_skip_enc[18] == '1': skip += skip_list[2]
        if self.R_skip_enc[19] == '1': skip += skip_list[3]
        if self.R_skip_enc[20] == '1': skip += skip_list[4]
        if self.R_skip_enc[-6] == '000000':
          skip = skip_list[-1]

    x = F.relu(skip)

    x = F.relu(self.end_conv_1(x))
    x = self.end_conv_2(x) # B, T, N, 1
    x = x.squeeze().permute(1,0,2) # TBN

    return x