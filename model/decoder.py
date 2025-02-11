import torch
import torch.nn as nn
import torch.nn.functional as FU

from model.utils import *

from collections import OrderedDict

class Decoder(nn.Module):
    def __init__(self, device, time_steps=91, feature_dim=256, head_num=4, k=4, F=6):
        super().__init__()
        self.device = device
        self.time_steps = time_steps                # T
        self.feature_dim = feature_dim              # D
        self.head_num = head_num                    # H
        self.k = k
        self.F = F

        onehots_ = torch.tensor(range(F)) # [F]
        self.onehots_ = FU.one_hot(onehots_, num_classes=F).to(self.device) # [F, F]

        self.layer_T = nn.Sequential(nn.Linear(self.feature_dim+self.F,feature_dim), nn.ReLU())

        self.layer_U = SelfAttLayer_dec(self.device,self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_V = SelfAttLayer_dec(self.device,self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)

        self.layer_W = SelfAttLayer_dec(self.device,self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_X = SelfAttLayer_dec(self.device,self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)

        self.layer_Y = nn.LayerNorm(self.feature_dim)

        self.layer_Z1 = nn.Sequential(nn.Linear(self.feature_dim,6), nn.ReLU(), Permute4Batchnorm((1,3,0,2)),
                            nn.BatchNorm2d(6), Permute4Batchnorm((2,0,3,1))) 
        self.layer_Z2 = nn.Linear(6,6)  # x, y, bbox_yaw, velocity_x, velocity_y, vel_yaw

    def forward(self, state_feat, batch_mask, padding_mask, hidden_mask=None):
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

        A,T,D = state_feat.shape
        assert (T==self.time_steps and D==self.feature_dim)

        # layer_R: [F, A, T, D]
        x = state_feat.unsqueeze(0).repeat(self.F,1,1,1)

        # layer_S: [F, F] -> [F, 1, 1, F] -> [F, A, T, F]
        onehots_ = self.onehots_.view(self.F,1,1,self.F).repeat(1,A,T,1)
        onehots_ = onehots_.to(state_feat.device)
        # [F, A, T, D] + [F, A, T, F] -> [F, A, T, D+F] # TODO why???
        x = torch.cat((x,onehots_),dim=-1)

        # layer_T: [F, A, T, D+F] -> [F, A, T, D]
        x = self.layer_T(x)

        # padding_mask and batch_mask is set to be opposite to nn.transformer and private encoder.
        # TODO : revert two masks and edit encoder following original code
        padding_mask = padding_mask==False
        batch_mask = batch_mask==False

        # [F, A, T, D] -> [F, A, T, D]
        x = self.layer_U(x,batch_mask=batch_mask, padding_mask=padding_mask)
        x = self.layer_V(x,batch_mask=batch_mask, padding_mask=padding_mask)
        
        x = self.layer_W(x,batch_mask=batch_mask, padding_mask=padding_mask)
        x = self.layer_X(x,batch_mask=batch_mask, padding_mask=padding_mask)

        # [F, A, T, D] -> [F, A, T, D] 
        x = self.layer_Y(x)
        # [F, A, T, D] -> [F, A, T, 6] # TODO z1 [score0 ,score1, score3, ...]
        x = self.layer_Z1(x)
        # TODO z2 [F, A, T, D] -> [F, A, T, 7] # x y z covx covy covz heading
        x = self.layer_Z2(x)

        return x
