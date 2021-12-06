import math
import os
import uuid
import time

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
# from IPython.display import HTML
import itertools
import torch
from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset

# from google.protobuf import text_format
# from waymo_open_dataset.metrics.ops import py_metrics_ops
# from waymo_open_dataset.metrics.python import config_util_py as config_util
# from waymo_open_dataset.protos import motion_metrics_pb2

# Example field definition
roadgraph_features = {
    'roadgraph_samples/dir':
        'float',
    'roadgraph_samples/id':
        'int',
    'roadgraph_samples/type':
        'int',
    'roadgraph_samples/valid':
        'int',
    'roadgraph_samples/xyz':
        'float',
}

# Features of other agents.
state_features = {
    'state/id':
        'float',
    'state/type':
        'float',
    'state/is_sdc':
        'int',
    'state/tracks_to_predict':
        'int',
    'state/current/bbox_yaw':
        'float',
    'state/current/height':
        'float',
    'state/current/length':
        'float',
    'state/current/timestamp_micros':
        'int',
    'state/current/valid':
        'int',
    'state/current/vel_yaw':
        'float',
    'state/current/velocity_x':
        'float',
    'state/current/velocity_y':
        'float',
    'state/current/width':
        'float',
    'state/current/x':
        'float',
    'state/current/y':
        'float',
    'state/current/z':
        'float',
    'state/future/bbox_yaw':
        'float',
    'state/future/height':
        'float',
    'state/future/length':
        'float',
    'state/future/timestamp_micros':
        'int',
    'state/future/valid':
        'int',
    'state/future/vel_yaw':
        'float',
    'state/future/velocity_x':
        'float',
    'state/future/velocity_y':
        'float',
    'state/future/width':
        'float',
    'state/future/x':
        'float',
    'state/future/y':
        'float',
    'state/future/z':
        'float',
    'state/past/bbox_yaw':
        'float',
    'state/past/height':
        'float',
    'state/past/length':
        'float',
    'state/past/timestamp_micros':
        'int',
    'state/past/valid':
        'int',
    'state/past/vel_yaw':
        'float',
    'state/past/velocity_x':
        'float',
    'state/past/velocity_y':
        'float',
    'state/past/width':
        'float',
    'state/past/x':
        'float',
    'state/past/y':
        'float',
    'state/past/z':
        'float',
}

traffic_light_features = {
    'traffic_light_state/current/state':
        'int',
    'traffic_light_state/current/valid':
        'int',
    'traffic_light_state/current/x':
        'float',
    'traffic_light_state/current/y':
        'float',
    'traffic_light_state/current/z':
        'float',
    'traffic_light_state/past/state':
        'int',
    'traffic_light_state/past/valid':
        'int',
    'traffic_light_state/past/x':
        'float',
    'traffic_light_state/past/y':
        'float',
    'traffic_light_state/past/z':
        'float',
    'traffic_light_state/future/state':
        'int',
    'traffic_light_state/future/valid':
        'int',
    'traffic_light_state/future/x':
        'float',
    'traffic_light_state/future/y':
        'float',
    'traffic_light_state/future/z':
        'float',
}

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)


# Example field definition
roadgraph_transforms = {
    'roadgraph_samples/dir':
        lambda x : np.reshape(x,(20000,3)),
    'roadgraph_samples/id':
        lambda x : np.reshape(x,(20000,1)),
    'roadgraph_samples/type':
        lambda x : np.reshape(x,(20000,1)),
    'roadgraph_samples/valid':
        lambda x : np.reshape(x,(20000,1)),
    'roadgraph_samples/xyz':
        lambda x : np.reshape(x,(20000,3)),
}

# Features of other agents.
state_transforms = {
    'state/id':
        lambda x : np.reshape(x,(128,)),
    'state/type':
        lambda x : np.reshape(x,(128,)),
    'state/is_sdc':
        lambda x : np.reshape(x,(128,)),
    'state/tracks_to_predict':
        lambda x : np.reshape(x,(128,)),
    'state/current/bbox_yaw':
        lambda x : np.reshape(x,(128,1)),
    'state/current/height':
        lambda x : np.reshape(x,(128,1)),
    'state/current/length':
        lambda x : np.reshape(x,(128,1)),
    'state/current/timestamp_micros':
        lambda x : np.reshape(x,(128,1)),
    'state/current/valid':
        lambda x : np.reshape(x,(128,1)),
    'state/current/vel_yaw':
        lambda x : np.reshape(x,(128,1)),
    'state/current/velocity_x':
        lambda x : np.reshape(x,(128,1)),
    'state/current/velocity_y':
        lambda x : np.reshape(x,(128,1)),
    'state/current/width':
        lambda x : np.reshape(x,(128,1)),
    'state/current/x':
        lambda x : np.reshape(x,(128,1)),
    'state/current/y':
        lambda x : np.reshape(x,(128,1)),
    'state/current/z':
        lambda x : np.reshape(x,(128,1)),
    'state/future/bbox_yaw':
        lambda x : np.reshape(x,(128,80)),
    'state/future/height':
        lambda x : np.reshape(x,(128,80)),
    'state/future/length':
        lambda x : np.reshape(x,(128,80)),
    'state/future/timestamp_micros':
        lambda x : np.reshape(x,(128,80)),
    'state/future/valid':
        lambda x : np.reshape(x,(128,80)),
    'state/future/vel_yaw':
        lambda x : np.reshape(x,(128,80)),
    'state/future/velocity_x':
        lambda x : np.reshape(x,(128,80)),
    'state/future/velocity_y':
        lambda x : np.reshape(x,(128,80)),
    'state/future/width':
        lambda x : np.reshape(x,(128,80)),
    'state/future/x':
        lambda x : np.reshape(x,(128,80)),
    'state/future/y':
        lambda x : np.reshape(x,(128,80)),
    'state/future/z':
        lambda x : np.reshape(x,(128,80)),
    'state/past/bbox_yaw':
        lambda x : np.reshape(x,(128,10)),
    'state/past/height':
        lambda x : np.reshape(x,(128,10)),
    'state/past/length':
        lambda x : np.reshape(x,(128,10)),
    'state/past/timestamp_micros':
        lambda x : np.reshape(x,(128,10)),
    'state/past/valid':
        lambda x : np.reshape(x,(128,10)),
    'state/past/vel_yaw':
        lambda x : np.reshape(x,(128,10)),
    'state/past/velocity_x':
        lambda x : np.reshape(x,(128,10)),
    'state/past/velocity_y':
        lambda x : np.reshape(x,(128,10)),
    'state/past/width':
        lambda x : np.reshape(x,(128,10)),
    'state/past/x':
        lambda x : np.reshape(x,(128,10)),
    'state/past/y':
        lambda x : np.reshape(x,(128,10)),
    'state/past/z':
        lambda x : np.reshape(x,(128,10)),
}

traffic_light_transforms = {
    'traffic_light_state/current/state':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/valid':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/x':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/y':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/z':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/past/state':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/valid':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/x':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/y':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/z':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/future/state':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/valid':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/x':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/y':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/z':
        lambda x : np.reshape(x,(80,16)),
}

features_transforms = {}
features_transforms.update(roadgraph_transforms)
features_transforms.update(state_transforms)
features_transforms.update(traffic_light_transforms)

def transform_func(feature):
    transform = features_transforms
    keys = transform.keys()
    for key in keys:
        func = transform[key]
        feat = feature[key]
        feature[key] = func(feat)
    return feature

def WaymoDataset(tfrecord_dir, idx_dir):

    tfrecord_pattern = tfrecord_dir+'/{}'
    index_pattern = idx_dir+'/{}'

    splits = {}
    fnlist = os.listdir(tfrecord_pattern.split('{}')[0])
    for fn in fnlist:
        splits[fn] = 1/len(fnlist)

    dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, splits, description=features_description, transform=transform_func, infinite=False)

    return dataset

# TODO : edit waymo dataset reading sequenial manner
# def WaymoSeqDataset(tfrecord_dir, idx_dir):
#     tfrecord_pattern = tfrecord_dir+'/{}'
#     index_pattern = idx_dir+'/{}'

#     splits = {}
#     fnlist = os.listdir(tfrecord_pattern.split('{}')[0])

def waymo_collate_fn(batch, GD=16, GS=1400): # GS = max number of static roadgraph element (1400), GD = max number of dynamic roadgraph (16)
    past_states_batch = np.array([]).reshape(-1,10,9)
    past_states_valid_batch = np.array([]).reshape(-1,10)
    current_states_batch = np.array([]).reshape(-1,1,9)
    current_states_valid_batch = np.array([]).reshape(-1,1)
    future_states_batch = np.array([]).reshape(-1,80,9)
    future_states_valid_batch = np.array([]).reshape(-1,80)
    states_batch = np.array([]).reshape(-1,91,9)

    states_padding_mask_batch = np.array([]).reshape(-1,91)
    states_hidden_mask_BP_batch = np.array([]).reshape(-1,91)
    states_hidden_mask_CBP_batch = np.array([]).reshape(-1,91)
    states_hidden_mask_GDP_batch =np.array([]).reshape(-1,91)

    roadgraph_feat_batch = np.array([]).reshape(-1,91,6)
    roadgraph_valid_batch = np.array([]).reshape(-1,91)

    traffic_light_feat_batch = np.array([]).reshape(-1,91,3)
    traffic_light_valid_batch = np.array([]).reshape(-1,91)

    num_agents = np.array([])

    for data in batch:
        # State of Agents
        # stack([128,10,], ...) -> [A, T, D] = [128, 10, 9]
        past_states = np.stack((data['state/past/x'],data['state/past/y'],data['state/past/bbox_yaw'],
                                    data['state/past/velocity_x'],data['state/past/velocity_y'],data['state/past/vel_yaw'],
                                        data['state/past/width'],data['state/past/length'],data['state/past/timestamp_micros']), axis=-1)
        # [128,10,] -> [A, T] = [128,10,] True/False
        past_states_valid = data['state/past/valid'] > 0.

        # stack([128,1,], ...) -> [A, T, D] = [128, 1, 9]
        current_states = np.stack((data['state/current/x'],data['state/current/y'],data['state/current/bbox_yaw'],
                                    data['state/current/velocity_x'],data['state/current/velocity_y'],data['state/current/vel_yaw'],
                                        data['state/current/width'],data['state/current/length'],data['state/current/timestamp_micros']), axis=-1)
        # [128,1,] -> [A, T] = [128,1,] True/False
        current_states_valid = data['state/current/valid'] > 0.

        # stack([128,80,], ...) -> [A, T, D] = [128, 80, 9]
        future_states = np.stack((data['state/future/x'],data['state/future/y'],data['state/future/bbox_yaw'],
                                    data['state/future/velocity_x'],data['state/future/velocity_y'],data['state/future/vel_yaw'],
                                        data['state/future/width'],data['state/future/length'],data['state/future/timestamp_micros']), axis=-1)
        # [128,80,] -> [A, T] = [128,80,] True/False
        future_states_valid = data['state/future/valid'] > 0.

        # [128, 10, 9],[128,1,9] [128,80,9] -> [128, 91, 9]    history:0-9 curr:10 future:11-90
        states_feat = np.concatenate((past_states,current_states,future_states),axis=1)
        # [128, 10],[128, 1],[128, 80] -> [128, 91] True/False
        states_valid = np.concatenate((past_states_valid,current_states_valid,future_states_valid),axis=1)

        # [128,] True/False
        states_any_mask = np.sum(states_valid,axis=1) > 0 # all 91 states of the "padding agent" is invalid
        # [N_valid, 91, 9]
        states_feat = states_feat[states_any_mask]
        # [N_valid, 91] True/False
        states_padding_mask = np.concatenate((past_states_valid[states_any_mask],current_states_valid[states_any_mask],future_states_valid[states_any_mask]), axis=1)
        
        # basic_mask = np.zeros((len(states_feat),91)).astype(np.bool_)
        # [N_valid, 91] True/False
        states_hidden_mask_BP = np.ones((len(states_feat),91)).astype(np.bool_)
        # 0-11 is not mask bug? why not :11 ?
        states_hidden_mask_BP[:,:12] = False 

        # [128, ] -> [N_valid, ]
        sdvidx = np.where(data['state/is_sdc'][states_any_mask] == 1)[0][0]
        # [N_valid, 91] True/False TODO bug ones?
        states_hidden_mask_CBP = np.zeros((len(states_feat),91)).astype(np.bool_)
        # TODO 0-11 is not mask, why not :11 ?
        states_hidden_mask_CBP[:,:12] = False
        # TODO 0-90 is not mask, why not sdvidx-1?
        states_hidden_mask_CBP[sdvidx-1,:] = False

        # [N_valid, 91] TODO bug ones?
        states_hidden_mask_GDP = np.zeros((len(states_feat),91)).astype(np.bool_)
        # TODO 0-11 is not mask, why not :11 ?
        states_hidden_mask_GDP[:,:12] = False
        # TODO 90 is not mask, why not sdvidx-1?
        states_hidden_mask_GDP[sdvidx-1,-1] = False
        # states_hidden_mask_CDP = np.zeros((len(states_feat),91)).astype(np.bool_)

        # [] -> [1] -> [2] -> [3]
        num_agents = np.append(num_agents, len(states_feat))
        
        # Static Road Graph
        # (20000,1) (20000,1) (20000,2) (20000,2) -> (20000, 6)
        roadgraph_feat = np.concatenate((data['roadgraph_samples/id'], data['roadgraph_samples/type'], 
                                            data['roadgraph_samples/xyz'][:,:2], data['roadgraph_samples/dir'][:,:2]), axis=-1)
        # [20000,1] -> [20000, 1] True/False
        roadgraph_valid = data['roadgraph_samples/valid'] > 0.
        valid_num = roadgraph_valid.sum()
        if valid_num > GS: #  GS=1400, GD=16,
            # [N_valid_GS, 6]
            roadgraph_feat = roadgraph_feat[roadgraph_valid[:,0]]
            spacing = valid_num // GS # e.g. 2000 // 1400 = 1  2900 // 1400 = 2 
            # [N_sample_GS, 6] N_sample_GS > GS
            roadgraph_feat = roadgraph_feat[::spacing, :]
            remove_num = len(roadgraph_feat) - GS
            # [N_sample_GS] True
            roadgraph_mask2 = np.full(len(roadgraph_feat), True)
            idx_remove = np.random.choice(range(len(roadgraph_feat)), remove_num, replace=False)
            # [N_sample_GS] True/False, there is GS element is True, N_sample_GS-GS is False
            roadgraph_mask2[idx_remove] = False
            # [GS, 6]
            roadgraph_feat = roadgraph_feat[roadgraph_mask2]
            # [GS, 1] True
            roadgraph_valid = np.ones((GS,1)).astype(np.bool_)
        else:
            # [N_valid_GS, 6],  N_valid_GS < GS
            roadgraph_feat = roadgraph_feat[roadgraph_valid[:,0]]
            # TODO bug? [N_valid_GS, 6] -> [GS, 6] ??

            # [GS, 1] False
            roadgraph_valid = np.zeros((GS,1)).astype(np.bool_)
            # [GS, 1] True/False 0:N_valid_GS is True, N_valid_GS:-1 is False
            roadgraph_valid[:valid_num,:] = True
            # (Optional) : construct roadgraph valid

        # [GS, 6]  -> [GS, 1, 6] -> [GS, 91, 6]
        roadgraph_feat = np.repeat(roadgraph_feat[:,np.newaxis,:],91,axis=1)
        # [GS, 1] -> [GS, 91]
        roadgraph_valid = np.repeat(roadgraph_valid,91,axis=1)

        # Dynamic Road Graph
        # [16, 10] [16, 10] [16, 10] -> [16, 10, 3]
        traffic_light_states_past = np.stack((data['traffic_light_state/past/state'].T,data['traffic_light_state/past/x'].T,data['traffic_light_state/past/y'].T),axis=-1)
        # [16, 10] True/False
        traffic_light_valid_past = data['traffic_light_state/past/valid'].T > 0.
        # [16, 1] [16, 1] [16, 1] -> [16, 1, 3]
        traffic_light_states_current = np.stack((data['traffic_light_state/current/state'].T,data['traffic_light_state/current/x'].T,data['traffic_light_state/current/y'].T),axis=-1)
        # [16, 1] True/False
        traffic_light_valid_current = data['traffic_light_state/current/valid'].T > 0.
        # [16, 80] [16, 80] [16, 80] -> [16, 80, 3]
        traffic_light_states_future = np.stack((data['traffic_light_state/future/state'].T,data['traffic_light_state/future/x'].T,data['traffic_light_state/future/y'].T),axis=-1)
        # [16, 80] True/False
        traffic_light_valid_future = data['traffic_light_state/future/valid'].T > 0.
        # [16, 10, 3] [16, 1, 3] [16, 80, 3] -> [16, 91, 3]
        traffic_light_feat = np.concatenate((traffic_light_states_past,traffic_light_states_current,traffic_light_states_future),axis=1)
        # [16, 10] [16, 1] [16, 80] -> [16, 91]
        traffic_light_valid = np.concatenate((traffic_light_valid_past,traffic_light_valid_current,traffic_light_valid_future),axis=1)

        # Concat across batch TODO why not concat between batch?
        # [-1,10,9] [128, 10, 9] -> [bs*128, 10, 9]
        past_states_batch = np.concatenate((past_states_batch, past_states), axis=0)
        # [-1,10] [128, 10] -> [bs*128, 10]
        past_states_valid_batch = np.concatenate((past_states_valid_batch, past_states_valid), axis=0)
        # [-1,1,9] [128, 1, 9] -> [bs*128, 1, 9]
        current_states_batch = np.concatenate((current_states_batch, current_states), axis=0)
        # [-1,1] [128, 1] -> [bs*128, 1]
        current_states_valid_batch = np.concatenate((current_states_valid_batch, current_states_valid), axis=0)
        # [-1,80,9] [128, 80, 9] -> [bs*128, 80, 9]
        future_states_batch = np.concatenate((future_states_batch, future_states), axis=0)
        # [-1,80] [128, 80] -> [bs*128, 80]
        future_states_valid_batch = np.concatenate((future_states_valid_batch, future_states_valid), axis=0)

        # [-1, 91, 9] [N_valid, 91, 9] -> [sumN, 91, 9]
        states_batch = np.concatenate((states_batch,states_feat), axis=0)
        # [-1, 91] [N_valid, 91] -> [sumN, 91]
        states_padding_mask_batch = np.concatenate((states_padding_mask_batch,states_padding_mask), axis=0)

        # [-1, 91] [N_valid, 91] -> [sumN, 91]
        states_hidden_mask_BP_batch = np.concatenate((states_hidden_mask_BP_batch,states_hidden_mask_BP), axis=0)
        # [-1, 91] [N_valid, 91] -> [sumN, 91]
        states_hidden_mask_CBP_batch = np.concatenate((states_hidden_mask_CBP_batch,states_hidden_mask_CBP), axis=0)
        # [-1, 91] [N_valid, 91] -> [sumN, 91]
        states_hidden_mask_GDP_batch =np.concatenate((states_hidden_mask_GDP_batch,states_hidden_mask_GDP), axis=0)

        # [-1, 91, 6] [GS, 91, 6] -> [bs*GS, 91, 6]
        roadgraph_feat_batch = np.concatenate((roadgraph_feat_batch, roadgraph_feat), axis=0)
        # [-1, 91] [GS, 91] -> [bs*GS, 91] 
        roadgraph_valid_batch = np.concatenate((roadgraph_valid_batch, roadgraph_valid), axis=0)

        # [-1, 91, 3] [GD, 91, 3] -> [bs*GD, 91, 3]
        traffic_light_feat_batch = np.concatenate((traffic_light_feat_batch, traffic_light_feat), axis=0)
        # [-1, 91] [GD, 91] -> [bs*GD, 91]
        traffic_light_valid_batch = np.concatenate((traffic_light_valid_batch, traffic_light_valid), axis=0)

    # num_agents [num1, num2, num3] -> [0, num1, num2, num3] -> [0, num1, num1+num2, num1+num2+num3=sumN]
    num_agents_accum = np.cumsum(np.insert(num_agents,0,0)).astype(np.int64)
    # [sumN, sumN] 0
    agents_batch_mask = np.zeros((num_agents_accum[-1],num_agents_accum[-1]))
    # [sumN, bs*GS] 0
    agent_rg_mask = np.zeros((num_agents_accum[-1],len(num_agents)*GS)) # bs = len(num_agents)
    # [sumN, bs*GD] 0
    agent_traffic_mask = np.zeros((num_agents_accum[-1],len(num_agents)*GD))

    # only allow interactive in the same batch: why not create a Batch dimention?????????
    for i in range(len(num_agents)):
        agents_batch_mask[num_agents_accum[i]:num_agents_accum[i+1], num_agents_accum[i]:num_agents_accum[i+1]] = 1 # agents in same batch
        agent_rg_mask[num_agents_accum[i]:num_agents_accum[i+1], GS*i:GS*(i+1)] = 1 # agent/rg in same batch
        agent_traffic_mask[num_agents_accum[i]:num_agents_accum[i+1], GD*i:GD*(i+1)] = 1 # agent/traffic in same batch

    # [sumN, 91, 9]
    states_batch = torch.FloatTensor(states_batch)
    # [sumN, 91]
    states_padding_mask_batch = torch.BoolTensor(states_padding_mask_batch) # True -> mask or not mask?
    # [sumN, 91]
    states_hidden_mask_BP_batch = torch.BoolTensor(states_hidden_mask_BP_batch)
    # [sumN, 91]
    states_hidden_mask_CBP_batch = torch.BoolTensor(states_hidden_mask_CBP_batch)
    # [sumN, 91]
    states_hidden_mask_GDP_batch = torch.BoolTensor(states_hidden_mask_GDP_batch)
    
    # [bs*GS, 91, 6]
    roadgraph_feat_batch = torch.FloatTensor(roadgraph_feat_batch)
    # [bs*GS, 91]
    roadgraph_valid_batch = torch.BoolTensor(roadgraph_valid_batch)
    # [bs*GD, 91, 3]
    traffic_light_feat_batch = torch.FloatTensor(traffic_light_feat_batch)
    # [bs*GD, 91]
    traffic_light_valid_batch = torch.BoolTensor(traffic_light_valid_batch)

    # [sumN, sumN]
    agents_batch_mask = torch.BoolTensor(agents_batch_mask)
    # [sumN, bs*GS]
    agent_rg_mask = torch.BoolTensor(agent_rg_mask)
    # [sumN, bs*GD] 0
    agent_traffic_mask = torch.BoolTensor(agent_traffic_mask)

    # TODO : agent-rg per batch mask (다른 scene 의 rg 와 agent 끼리 attention = 0  으로 만들기 위함)
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
    return (states_batch, agents_batch_mask, states_padding_mask_batch, 
                (states_hidden_mask_BP_batch, states_hidden_mask_CBP_batch, states_hidden_mask_GDP_batch), 
                    roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch,
                        agent_rg_mask, agent_traffic_mask)
