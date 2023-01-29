
rg_epochs = [100] 
rg_early_stop = [100] 
rg_lr = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
rg_lr_decay_steps = [100]
rg_lr_decay_ratio = [1]
rg_batch_size = [8, 16, 32, 64, 128]
rg_grad_norm = [0, 1, 3, 5, 7, 9]
rg_optim = ['sgd', 'rmsprop', 'adam', 'adamw', 'adamax']
rg_wdecay = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]
rg_loss = ['mae'] # 'mse'
rg_dropout = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
rg_mid_channel = [64,32,16]
rg_skip_channel = [64,32,16]
rg_end_channel = [256,128,64,32]
rg_cl_steps = [None, 3, 5, 7]

# G
rg_G_method = ['mtgnn']
rg_G_k = [5, 10, 20, 40, 60]
rg_G_dim = [10, 20, 40, 60, 80]
rg_G_alpha = [1,2,3]
rg_G_mix = [0, 0.25, 0.5, 0.75, 1]

# S
rg_S_hop = [1,2,3]
rg_S_fusion = ['concat']
rg_S_hopalpha = [0.05]

# T
rg_T_dilation = [2,1]
rg_T_ks_len = [1,2,4,8]
rg_T_ks = [2,3,4,5,6,7,8,9,10,11,12] 

# R
rg_R_skip = [0,1,2,3] 
rg_R_residual = [0,1,2,3]