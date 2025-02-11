import torch
import torch.nn as nn

from model.utils import *

from collections import OrderedDict

class Encoder(nn.Module):
    def __init__(self, device, in_feat_dim, in_dynamic_rg_dim, in_static_rg_dim, time_steps=91, feature_dim=256, 
    head_num=4, max_dynamic_rg=16, max_static_rg=1400, k=4): # what is k?
        super().__init__()
        self.device = device
        self.time_steps = time_steps                # T
        self.feature_dim = feature_dim              # D
        self.head_num = head_num                    # H
        self.max_dynamic_rg = max_dynamic_rg        # GD
        self.max_static_rg = max_static_rg          # GS
        assert feature_dim % head_num == 0      
        self.head_dim = int(feature_dim/head_num)   # d
        self.k = k                                  # k

        # layer A : input -> [A,T,in_feat_dim] / output -> [A,T,D]
        self.layer_A = nn.Sequential(nn.Linear(in_feat_dim,feature_dim), nn.ReLU(), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(feature_dim), Permute4Batchnorm((0,2,1)))
        # layer B : input -> [Gd,T,in_dynamic_rg_dim] / output -> [Gd,T,D]
        self.layer_B = nn.Sequential(nn.Linear(in_dynamic_rg_dim,feature_dim), nn.ReLU(), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(feature_dim), Permute4Batchnorm((0,2,1)))
        # layer C : input -> [Gs,T,in_static_rg_dim] / output -> [Gs,T,D]
        self.layer_C = nn.Sequential(nn.Linear(in_static_rg_dim,feature_dim), nn.ReLU(), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(feature_dim), Permute4Batchnorm((0,2,1)))
        # layer D,E,F,G,H,I : input -> [A,T,D] / outpu -> [A,T,D]
        self.layer_D = SelfAttLayer(self.device, self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_E = SelfAttLayer(self.device, self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)
        self.layer_F = SelfAttLayer(self.device, self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_G = SelfAttLayer(self.device, self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)
        self.layer_H = SelfAttLayer(self.device, self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_I = SelfAttLayer(self.device, self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)

        self.layer_J = CrossAttLayer(self.device, self.time_steps,self.feature_dim,self.head_num,self.k)
        self.layer_K = CrossAttLayer(self.device, self.time_steps,self.feature_dim,self.head_num,self.k)

        self.layer_L = SelfAttLayer(self.device, self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_M = SelfAttLayer(self.device, self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)

        self.layer_N = CrossAttLayer(self.device, self.time_steps,self.feature_dim,self.head_num,self.k)
        self.layer_O = CrossAttLayer(self.device, self.time_steps,self.feature_dim,self.head_num,self.k)

        self.layer_P = SelfAttLayer(self.device, self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_Q = SelfAttLayer(self.device, self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)
        

    def forward(self, state_feat, agent_batch_mask, padding_mask, hidden_mask, 
                    road_feat, roadgraph_valid, traffic_light_feat, traffic_light_valid,
                        agent_rg_mask, agent_traffic_mask):
        '''
        # states_batch [sumN', 91, 9]
        # agents_batch_mask [sumN', sumN']
        # states_padding_mask_batch [sumN', 91]
        # states_hidden_mask_batch [sumN', 91]
        # roadgraph_feat_batch [bs*GS, 91, 6]
        # roadgraph_valid_batch [bs*GS, 91]
        # traffic_light_feat_batch [bs*GD, 91, 3]
        # traffic_light_valid_batch [bs*GD, 91]
        # agent_rg_mask [sumN', bs*GS]
        # agent_traffic_mask [sumN', bs*GD]
        '''

        # TODO agent_batch_mask [A, A] padding_mask[A, T] hidden_mask? roadgraph_valid? traffic_light_valid? agent_traffic_mask?
        # hidden_mask True:not_masked False:masked
        state_feat[hidden_mask==False] = -1
        # [A, T, in_feat_dim] -> [A, T, D]
        A_ = self.layer_A(state_feat)
        # [Gd, T, in_dynamic_rg_feat_dim] -> [Gd, T, D]
        B_ = self.layer_B(traffic_light_feat)
        # [Gs, T, in_static_rg_feat_dim] -> [Gs, T, D]
        C_ = self.layer_C(road_feat)
        
        output,_,_,_ = self.layer_D(A_,agent_batch_mask, padding_mask, hidden_mask)
        output,_,_,_ = self.layer_E(output,agent_batch_mask, padding_mask, hidden_mask)
        output,_,_,_ = self.layer_F(output,agent_batch_mask, padding_mask, hidden_mask)
        output,_,_,_ = self.layer_G(output,agent_batch_mask, padding_mask, hidden_mask)
        output,_,_,_ = self.layer_H(output,agent_batch_mask, padding_mask, hidden_mask)
        output,_,_,_ = self.layer_I(output,agent_batch_mask, padding_mask, hidden_mask)

        # TODO : add additional artificial agent/time AND adjust mask for it

        output,_,_,_ = self.layer_J(output,C_,agent_rg_mask, padding_mask, roadgraph_valid)
        output,_,_,_ = self.layer_K(output,B_,agent_traffic_mask, padding_mask, traffic_light_valid)

        output,_,_,_ = self.layer_L(output,agent_batch_mask, padding_mask, hidden_mask)
        output,_,_,_ = self.layer_M(output,agent_batch_mask, padding_mask, hidden_mask)

        output,_,_,_ = self.layer_N(output,C_,agent_rg_mask, padding_mask, roadgraph_valid)
        output,_,_,_ = self.layer_O(output,B_,agent_traffic_mask, padding_mask, traffic_light_valid)

        output,_,_,_ = self.layer_P(output,agent_batch_mask, padding_mask, hidden_mask)
        Q_,_,_,_ = self.layer_Q(output,agent_batch_mask, padding_mask, hidden_mask)

        return Q_, agent_batch_mask, padding_mask



