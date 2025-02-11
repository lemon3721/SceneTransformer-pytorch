import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Permute4Batchnorm(nn.Module):
    def __init__(self,order):
        super(Permute4Batchnorm, self).__init__()
        self.order = order
    
    def forward(self, x):
        assert len(self.order) == len(x.shape)
        return x.permute(self.order)

class ScaleLayer(nn.Module):

   def __init__(self, shape, init_value=1e-3):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor(shape).fill_(init_value))

   def forward(self, input):
       return input * self.scale

class SelfAttLayer(nn.Module):
    def __init__(self, device, time_steps=91, feature_dim=256, head_num=4, k=4, across_time=True):
        super().__init__()

        self.viewmodule_ = View((-1,time_steps,head_num, int(feature_dim/head_num)))
        self.layer_X_ = nn.LayerNorm(feature_dim)
        self.layer_K_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_V_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_Q0_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_Q_ = ScaleLayer(int(feature_dim/head_num)) # [d]

        # TODO bug? should be feature_dim/head_num
        self.scale = torch.sqrt(torch.FloatTensor([head_num])).to(device)

        self.layer_Y2_ = nn.Sequential(View((-1,time_steps,feature_dim)), nn.Linear(feature_dim,feature_dim), nn.ReLU())
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,k*feature_dim), nn.ReLU())
        self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
        self.layer_Z_ = nn.LayerNorm(feature_dim)

        self.across_time = across_time

    def forward(self, x, batch_mask, padding_mask=None, hidden_mask=None):
        # batch_mask [A, A]
        # padding_mask [A, T]

        # [A, T, D] -> [A, T, D]
        x = self.layer_X_(x)
        # [A, T, D] -> [A, T, H, d] # TODO bug? multi head is generate by different K V Q
        K = self.layer_K_(x)
        # [A, T, D] -> [A, T, H, d] # TODO bug? multi head is generate by different K V Q
        V = self.layer_V_(x)
        # [A, T, D] -> [A, T, H, d] # TODO bug? multi head is generate by different K V Q
        Q0 = self.layer_Q0_(x)
        # [A, T, H, d] * [d] -> [A, T, H, d]
        Q = self.layer_Q_(Q0)    # Q,K,V -> [A,T,H,d]
        
        self.scale = self.scale.to(K.device)

        if self.across_time:
            # Q,K,V: [A,T,H,d] -> [A,H,T,d] 
            Q, K, V = Q.permute(0,2,1,3), K.permute(0,2,1,3), V.permute(0,2,1,3)    
            # [A, H, Tq, d] * [A, H, d, Tk] -> [A, H, Tq, Tk]
            energy = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale                               # [A,H,T,T]
            # [A, H, Tq, Tk] -> [A, Tk, H, Tq]
            energy.permute(0,3,1,2)[padding_mask==False] = -1e10
            # [A, H, Tq, Tk] -> [A, Tq, H, Tk]
            energy.permute(0,2,1,3)[padding_mask==False] = -1e10
            # [A, H, Tq, Tk]
            attention = torch.softmax(energy, dim=-1)                               # [A,H,T,T]
            # [A, H, Tq, Tk] * [A, H, Tk, d] -> [A, H, Tq, d]
            Y1_ = torch.matmul(attention, V)                                        # [A,H,T,d]
            # [A, H, Tq, d] -> [A, Tq, H, d]
            Y1_ = Y1_.permute(0,2,1,3).contiguous()                                 # [A,T,H,d]

        else:
            # Q,K,V [A,T,H,d] -> [T,H,A,d]
            Q, K, V = Q.permute(1,2,0,3), K.permute(1,2,0,3), V.permute(1,2,0,3)    
            # [T, H, Aq, d] * [T, H, d, Ak] -> [T, H, Aq, Ak]
            energy = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale                               # [T,H,A,A]

            # if batch_mask is not None:                                              # batch_mask -> [A,A]
            # [T, H, Aq, Ak] -> batch_mask [A, A] TODO
            energy = energy.masked_fill(batch_mask==0, -1e10)   # 0 for ignoring attention
            # [T, H, Aq, Ak] -> [Aq, T, H, Ak] padding_mask:[A, T]
            energy.permute(2,0,1,3)[padding_mask==False] = -1e10
            # [T, H, Aq, Ak] -> [Ak, T, H, Aq]
            energy.permute(3,0,1,2)[padding_mask==False] = -1e10

            # [T, H, Aq, Ak]
            attention = torch.softmax(energy, dim=-1)                               # [T,H,A,A]

            # [T, H, Aq, Ak] * [T, H, Ak, d] -> [T, H, Aq, d]
            Y1_ = torch.matmul(attention, V)                                        # [T,H,A,d]
            # [T, H, Aq, d] -> [Aq, T, H, d]
            Y1_ = Y1_.permute(2,0,1,3).contiguous()                                 # [A,T,H,d]

        # [A, T, H, d] -> [A, T, D] # TODO bug? Multi head [d] -> [D] and merge? 
        Y2_ = self.layer_Y2_(Y1_)
        # [A, T, D] + [A, T, D] -> [A, T, D]
        S_ = Y2_ + x
        # [A, T, D] -> [A, T, kD]
        F1_ = self.layer_F1_(S_)
        # [A, T, kD] -> [A, T, D] 
        F2_ = self.layer_F2_(F1_)
        # [A, T, D] -> [A, T, D] 
        Z_ = self.layer_Z_(F2_)

        return Z_, Q, K, V # -> [A,T,D], [A,T,H,d]*3

class CrossAttLayer(nn.Module):
    def __init__(self, device, time_steps=91, feature_dim=256, head_num=4, k=4):
        super().__init__()

        self.viewmodule_ = View((-1,time_steps,head_num, int(feature_dim/head_num)))
        self.layer_X_ = nn.LayerNorm(feature_dim)

        self.layer_K_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_V_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_Q0_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_Q_ = ScaleLayer(int(feature_dim/head_num))

        # TODO bug? should be feature_dim/head_num
        self.scale = torch.sqrt(torch.FloatTensor([head_num])).to(device)

        self.layer_Y2_ = nn.Sequential(View((-1,time_steps,feature_dim)), nn.Linear(feature_dim,feature_dim), nn.ReLU())
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,k*feature_dim), nn.ReLU())
        self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
        self.layer_Z_ = nn.LayerNorm(feature_dim)

    def forward(self, agent, rg, agent_rg_mask, padding_mask, rg_valid_mask): # agent -> [A,T,D] / rg -> [G,T,D]
        # [A, T, D] -> [A, T, D]
        agent = self.layer_X_(agent)
        # [G, T, D] -> [G, T, D]
        rg = self.layer_X_(rg)

        # [G, T, D] -> [G, T, H, d] # TODO bug? multi head is generate by different K V Q 
        K = self.layer_K_(rg)                                                       # [G,T,H,d]
        # [G, T, D] -> [G, T, H, d] # TODO bug? multi head is generate by different K V Q 
        V = self.layer_V_(rg)                                                       # [G,T,H,d]
        # [A, T, D] -> [A, T, H, d] # TODO bug? multi head is generate by different K V Q 
        Q0 = self.layer_Q0_(agent)
        # [A, T, H, d] * [d] -> [A, T, H, d]
        Q = self.layer_Q_(Q0)                                                       # [A,T,H,d]

        # Q,K,V [G/A, T, H, d] -> [T, H, G/A, d]
        Q, K, V = Q.permute(1,2,0,3), K.permute(1,2,0,3), V.permute(1,2,0,3)    # Q -> [T,H,A,d] / K,V -> [T,H,G,d]
        # [T, H, A, d] * [T, H, d, G] -> [T, H, A, G]
        energy = torch.matmul(Q,K.permute(0,1,3,2)) / Q.shape[1]               # [T,H,A,G]

        # [T, H, A, G] -> [A, G, T, H]
        energy.permute(2,3,0,1)[agent_rg_mask==False] = -1e10
        # [T, H, A, G] -> [A, T, H, G]
        energy.permute(2,0,1,3)[padding_mask==False] = -1e10
        # [T, H, A, G] -> [G, T, H, A]
        energy.permute(3,0,1,2)[rg_valid_mask==False] = -1e10

        # [T, H, A, G]
        attention = torch.softmax(energy, dim=-1)                               # [T,H,A,G]
        # [T, H, A, G] * [T, H, G, d] -> [T, H, A, d]
        Y1_ = torch.matmul(attention, V)                                        # [T,H,A,d]
        # [A, T, H, d]
        Y1_ = Y1_.permute(2,0,1,3).contiguous()                                 # [A,T,H,d]

        # [A, T, H, d] -> [A, T, D] # TODO multi head [d] -> [D] and merge?
        Y2_ = self.layer_Y2_(Y1_)
        # [A, T, D] + [A, T, D] -> [A, T, D]
        S_ = Y2_ + agent
        # [A, T, D] -> [A, T, kD]
        F1_ = self.layer_F1_(S_)
        # [A, T, kD] -> [A, T, D]
        F2_ = self.layer_F2_(F1_)
        # [A, T, D] -> [A, T, D]
        Z_ = self.layer_Z_(F2_)

        return Z_, Q, K, V # -> [A,T,D], [G,T,H,d], [A,T,H,d]*2


class SelfAttLayer_dec(nn.Module):
    def __init__(self, device, time_steps=91, feature_dim=256, head_num=4, k=4, across_time=True):
        super().__init__()
        self.device = device
        self.across_time = across_time
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.k = k

        self.layer_X_ = nn.LayerNorm(feature_dim)
        self.layer_att_ = nn.MultiheadAttention(embed_dim=feature_dim,num_heads=head_num)
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,k*feature_dim), nn.ReLU())
        self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
        self.layer_Z_ = nn.LayerNorm(feature_dim)

    def forward(self, x, batch_mask, padding_mask=None, hidden_mask=None):
        F,A,T,D = x.shape
        assert (T==self.time_steps and D==self.feature_dim)
        A,A = batch_mask.shape
        A,T = padding_mask.shape
        assert T==self.time_steps

        # [F, A, T, D] -> [F, A, T, D]
        x_ = self.layer_X_(x) # [F,A,T,D]

        if self.across_time:
            # [F, A, T, D] -> [F*A, T, D] -> [T, F*A, D]
            q = x_.reshape((-1,T,D)).permute(1,0,2) # batch_first=False [L, N(bs), E] = [T, F*A, D]
            # [T, F*A, D]
            k,v = q.clone(),q.clone()

            # [A, T] -> [F*A, T]
            key_padding_mask = padding_mask.repeat(F,1)
            attn_mask = None  
            # att_output : q:[T, F*A, D], k:[T, F*A, D], v:[T, F*A, D] key_padding_mask:[F*A, T] -> [T,F*A,D]
            att_output, _ = self.layer_att_(q,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            # att_output : [T, F*A, D] -> [T, F, A, D] -> [F,A,T,D]
            att_output = att_output.reshape((T,F,A,D)).permute(1,2,0,3)
        else:
            # [F, A, T, D] -> [F, T, A, D] -> [F*T, A, D] -> [A, F*T, D]
            q = x_.permute(0,2,1,3).reshape((-1,A,D)).permute(1,0,2)     # [L,N,E] = [A,T*F,D] = [546,128,256]
            # [A, F*T, D]
            k, v = q.clone(), q.clone()                 # [S,N,E] = [A,T*F,D]

            # [A, T] -> [T, A] -> [F*T, A]
            key_padding_mask = padding_mask.permute(1,0).repeat(F,1) # [N,S] = [T*F,A]
            # [A, A]
            attn_mask = batch_mask                      # [L,S] = [A,A]
            # att_output : q:[A, F*T, D] k:[A, F*T, D] v:[A, F*T, D] key_padding_mask:[F*T, A] -> [A,F*T,D]
            att_output, _ = self.layer_att_(q,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            # att_output : [A, F*T, D] -> [A, F, T, D] -> [F,A,T,D]
            att_output = att_output.reshape((A,F,T,D)).permute(1,0,2,3)

        # [F,A,T,D] + [F,A,T,D] -> [F,A,T,D]
        S_ = att_output + x
        # [F, A, T, D] -> [F, A, T, kD]
        F1_ = self.layer_F1_(S_)
        # [F, A, T, kD] -> [F, A, T, D]
        F2_ = self.layer_F2_(F1_)
        # [F, A, T, D] -> [F, A, T, D]
        Z_ = self.layer_Z_(F2_)

        return Z_