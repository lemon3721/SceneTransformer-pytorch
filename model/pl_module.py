import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.functional as FU

import pytorch_lightning as pl

from model.encoder import Encoder
from model.decoder import Decoder

class SceneTransformer(pl.LightningModule):
    def __init__(self, device, in_feat_dim, in_dynamic_rg_dim, in_static_rg_dim, 
                    time_steps, feature_dim, head_num, k, F):
        super(SceneTransformer, self).__init__()
        # self.device = device
        self.in_feat_dim = in_feat_dim
        self.in_dynamic_rg_dim = in_dynamic_rg_dim
        self.in_static_rg_dim = in_static_rg_dim
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.k = k
        self.F = F

        self.encoder = Encoder(self.device, self.in_feat_dim, self.in_dynamic_rg_dim, self.in_static_rg_dim,
                                    self.time_steps, self.feature_dim, self.head_num)
        self.decoder = Decoder(self.device, self.time_steps, self.feature_dim, self.head_num, self.k, self.F)
        # self.model = nn.Sequential(self.encoder, self.decoder)
        # self.model = self.model.to(self.device)
        
    def forward(self, states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch,
                    roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch,
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

        encodings,_,_ = self.encoder(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch,
                                    roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch,
                                        agent_rg_mask, agent_traffic_mask)
        # [F, A, T, 6] 
        decoding = self.decoder(encodings, agents_batch_mask, states_padding_mask_batch)
        # [F, A, T, 6] -> [A, T, F, 6] 
        return decoding.permute(1,2,0,3)

    def training_step(self, batch, batch_idx):
        # states_batch [sumN, 91, 9]    
        # agents_batch_mask [sumN, sumN]
        # states_padding_mask_batch [sumN, 91] 
        # states_hidden_mask_BP_batch [sumN, 91]
        # states_hidden_mask_CBP_batch [sumN, 91]
        # states_hidden_mask_GDP_batch [sumN, 91]
        # roadgraph_feat_batch [bs*GS, 91, 6]
        # roadgraph_valid_batch [bs*GS, 91]
        # traffic_light_feat_batch [bs*GD, 91, 3]
        # traffic_light_valid_batch [bs*GD, 91]
        # agent_rg_mask [sumN, bs*GS]
        # agent_traffic_mask [sumN, bs*GD]
        states_batch, agents_batch_mask, states_padding_mask_batch, \
                (states_hidden_mask_BP, states_hidden_mask_CBP, states_hidden_mask_GDP), \
                    roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch, \
                        agent_rg_mask, agent_traffic_mask = batch

        # states_batch, agents_batch_mask, states_padding_mask_batch, \
        #         (states_hidden_mask_BP, states_hidden_mask_CBP, states_hidden_mask_GDP), \
        #             roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch, \
        #                 agent_rg_mask, agent_traffic_mask = states_batch.to(self.device), agents_batch_mask.to(self.device), states_padding_mask_batch.to(self.device), \
        #                                                                 (states_hidden_mask_BP.to(self.device), states_hidden_mask_CBP.to(self.device), states_hidden_mask_GDP.to(self.device)), \
        #                                                                     roadgraph_feat_batch.to(self.device), roadgraph_valid_batch.to(self.device), traffic_light_feat_batch.to(self.device), traffic_light_valid_batch.to(self.device), \
        #                                                                         agent_rg_mask.to(self.device), agent_traffic_mask.to(self.device)
        
        # TODO : randomly select hidden mask
        # states_hidden_mask_batch [sumN, 91]
        states_hidden_mask_batch = states_hidden_mask_BP
        
        # states_padding_mask_batch(sunN, 91) 0:pad 1: not pad,     
        # states_hidden_mask_batch(sumN, 91) False:not mask  True:mask
        # [sumN] True/False
        no_nonpad_mask = torch.sum((states_padding_mask_batch*~states_hidden_mask_batch),dim=-1) != 0
        # states_batch [sumN, 91, 9] -> [sumN', 91, 9]
        states_batch = states_batch[no_nonpad_mask]
        # agents_batch_mask [sumN, sumN] -> [sumN', sumN']
        agents_batch_mask = agents_batch_mask[no_nonpad_mask][:,no_nonpad_mask]
        # states_padding_mask_batch [sumN, 91] -> [sumN', 91]
        states_padding_mask_batch = states_padding_mask_batch[no_nonpad_mask]
        # states_hidden_mask_batch [sumN, 91] -> [sumN', 91]
        states_hidden_mask_batch = states_hidden_mask_batch[no_nonpad_mask]
        # agent_rg_mask [sumN, bs*GS] -> [sumN', bs*GS]
        agent_rg_mask = agent_rg_mask[no_nonpad_mask]
        # agent_traffic_mask [sumN, bs*GD] -> [sumN', bs*GD]
        agent_traffic_mask = agent_traffic_mask[no_nonpad_mask]                                                                    
        
        # Predict
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
        # 
        # -> prediction: [A, T, F, 6] = [sumN', 91, F, 6] 
        prediction = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, 
                        roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch,
                            agent_rg_mask, agent_traffic_mask)

        # Calculate Loss
        # [sumN', 91] * [sumN', 91],  sumN' is all valid but 91 have some invalid element
        to_predict_mask = states_padding_mask_batch*states_hidden_mask_batch
        # [sumN', 91, in_feat_dim] -> [sumN', 91, 6] x y yaw vx vy vyaw 
        gt = states_batch[:,:,:6][to_predict_mask]
        # [A, T, F, 6] = [sumN', 91, F, 6] -> [sumN', 91, F, 6]
        prediction = prediction[to_predict_mask]    
        
        Loss = nn.MSELoss(reduction='none')
        # [sumN', 91, 6] ->bug? unsqueeze(-2).repeat(1,1,F, 1)-> [sumN', 91, F, 6]
        loss_ = Loss(gt.unsqueeze(1).repeat(1,6,1), prediction)
        # [sumN', 91, F, 6] -> [F, sumN', 91, 6] -> [F, sumN']
        loss_ = torch.min(torch.sum(torch.sum(loss_, dim=0),dim=-1))

        return loss_

    def validation_step(self, batch, batch_idx):
        states_batch, agents_batch_mask, states_padding_mask_batch, \
                (states_hidden_mask_BP, states_hidden_mask_CBP, states_hidden_mask_GDP), \
                    roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch, \
                        agent_rg_mask, agent_traffic_mask = batch

        # states_batch, agents_batch_mask, states_padding_mask_batch, \
        #         (states_hidden_mask_BP, states_hidden_mask_CBP, states_hidden_mask_GDP), \
        #             roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch, \
        #                 agent_rg_mask, agent_traffic_mask = states_batch.to(self.device), agents_batch_mask.to(self.device), states_padding_mask_batch.to(self.device), \
        #                                                                 (states_hidden_mask_BP.to(self.device), states_hidden_mask_CBP.to(self.device), states_hidden_mask_GDP.to(self.device)), \
        #                                                                     roadgraph_feat_batch.to(self.device), roadgraph_valid_batch.to(self.device), traffic_light_feat_batch.to(self.device), traffic_light_valid_batch.to(self.device), \
        #                                                                         agent_rg_mask.to(self.device), agent_traffic_mask.to(self.device)
        
        # TODO : randomly select hidden mask
        states_hidden_mask_batch = states_hidden_mask_BP
        
        no_nonpad_mask = torch.sum((states_padding_mask_batch*~states_hidden_mask_batch),dim=-1) != 0
        states_batch = states_batch[no_nonpad_mask]
        agents_batch_mask = agents_batch_mask[no_nonpad_mask][:,no_nonpad_mask]
        states_padding_mask_batch = states_padding_mask_batch[no_nonpad_mask]
        states_hidden_mask_batch = states_hidden_mask_batch[no_nonpad_mask]
        agent_rg_mask = agent_rg_mask[no_nonpad_mask]
        agent_traffic_mask = agent_traffic_mask[no_nonpad_mask]                                                                    
        
        # Predict
        prediction = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, 
                        roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch,
                            agent_rg_mask, agent_traffic_mask)

        # Calculate Loss
        to_predict_mask = states_padding_mask_batch*states_hidden_mask_batch
        
        gt = states_batch[:,:,:6][to_predict_mask]
        prediction = prediction[to_predict_mask]     
        
        Loss = nn.MSELoss(reduction='none')
        loss_ = Loss(gt.unsqueeze(1).repeat(1,6,1), prediction)
        loss_ = torch.min(torch.sum(torch.sum(loss_, dim=0),dim=-1))

        self.log_dict({'val_loss': loss_})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
