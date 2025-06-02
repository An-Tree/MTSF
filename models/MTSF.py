import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

class moving_avg(nn.Module):
    
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x): 
        
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)) 
        return x


class series_decomp(nn.Module):
    
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x): 
        x = x.permute(0, 2, 1)
        moving_mean = self.moving_avg(x)
        return moving_mean 




class Model_backbone(nn.Module):
    def __init__(self, xp_i, kernel_size, m, r, lambda_, beta,
                 m_layers, c_in:int, seq_len:int, pred_len:int, patch_len:int=24, stride:int=24, n_layers:int=3, d_model=128, n_heads=16, d_ff:int=256, 
                 attn_dropout:float=0., dropout:float=0., res_attention:bool=True, store_attn:bool=False, padding_patch = None, **kwargs):
        
        super().__init__()
        
        self.xp_i = xp_i
        if self.xp_i:
            self.xp_forecast = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(c_in)]) 
        else:
            self.xp_forecast = nn.Linear(seq_len, pred_len) 

        self.LP = LP(kernel_size=kernel_size)
        self.avg_dec = series_decomp(kernel_size)
        
        self.m = m
        self.r = r
        self.lambda_ = lambda_

        self.inter_backbone = channel_att_block(beta, c_in, seq_len, d_model, d_ff, n_heads, dropout, m_layers)
        self.intra_backbone = patch_att_block(seq_len, pred_len, d_model, n_layers, n_heads, d_ff, patch_len=patch_len, stride=stride,
                                              dropout=dropout, padding_patch=padding_patch)

        self.n_vars = c_in
        self.pred_len = pred_len

        self.proj = Projection(d_model, patch_len)
    
    def forward(self, z):  
        
        xp = self.LP(z)
        res = z - xp
        if self.xp_i:
            x_pred = torch.stack([self.xp_forecast[i](xp[:,i,:]) for i in range(self.n_vars)], dim=1)
        else:
            x_pred = self.xp_forecast(xp)
        ApEn = self.approximate_entropy(xp, self.avg_dec(z), m=self.m, r=self.r, lambda_=self.lambda_)

        z_inter, kl_loss = self.inter_backbone(res,res) 
        
        result = self.intra_backbone(z_inter, res, x_pred)  
                                 
        return result, ApEn, kl_loss

    def approximate_entropy(self, U, O, m, r, lambda_ = 0.2): 
        
        batch_size, nvars, N = U.shape
        if r is None:
            r = 0.3 * torch.std(U, dim=2, keepdim=True)  
        else:
            r = torch.tensor(r, dtype=U.dtype, device=U.device)  
            r = r.view(1, 1, 1)  
            r = r.repeat(U.shape[0], U.shape[1], 1)  
        
        phi_m = self._phi(U, m, r)
        phi_m1 = self._phi(O, m, r)

        return torch.mean(phi_m1 - phi_m)*lambda_  

    def _phi(self, X, m, r):
        N = X.shape[2]
        
        X_m = X.unfold(dimension=2, size=m, step=1)  
        X_m = X_m.contiguous()

        diff = torch.abs(X_m.unsqueeze(3) - X_m.unsqueeze(2))  
        dist = torch.max(diff, dim=-1).values  

        softness = 100.0  
        C = torch.sum(torch.sigmoid((r.unsqueeze(2) - dist) * softness), dim=-1)        
        C = C / (N - m + 1)

        return torch.mean(torch.log(C + 1e-8), dim=-1)  


class LP(nn.Module):
    def __init__(self,kernel_size=25):
        super(LP, self).__init__()        
        s = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size): s[0, 0, i] = math.exp(-((i - kernel_size // 2) / (2 * 1.0)) ** 2)
        self.dec = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int(kernel_size//2), padding_mode='replicate', bias=True)
        self.dec.weight.data = F.softmax(s,dim=-1)
        self.dec.bias.data.fill_(0.0)
        
    def forward(self, inp):
        input_channels = torch.split(inp, 1, dim=1)
        conv_outputs = [self.dec(input_channel) for input_channel in input_channels]
        out = torch.cat(conv_outputs, dim=1)
        return out

class Projection(nn.Module):
    def __init__(self, d_model, patch_len):
        super().__init__()
        self.linear = nn.Linear(d_model,patch_len)
        self.flatten = nn.Flatten(start_dim = -2)
            
    def forward(self, x):                         
        x = self.linear(x)
        x = self.flatten(x)
        return x

class channel_att_block(nn.Module):
    def __init__(self, beta, dec_in, seq_len, d_model, d_ff, n_heads, dropout, m_layers):
        super(channel_att_block, self).__init__()
        self.embed = nn.Linear(seq_len, d_model)

        self.cross_att = nn.ModuleList([
            Channel_Attention_Layer(dec_in, d_model, d_ff, n_heads, dropout, attn_dropout=0.)
            for _ in range(m_layers)
        ])

        self.kl_loss = D_kl(feature_dim=d_model, z_dim=d_model, num_channels=dec_in, beta=beta)
        self.m_layers = m_layers

    def forward(self, q, inp): 

        q = self.embed(q)
        inp = self.embed(inp) 

        kl = 0
        for mod in self.cross_att: 
            inp = mod(q, inp, inp)
            kl += self.kl_loss(inp) 
        
        return inp , kl/self.m_layers 

class Channel_Attention_Layer(nn.Module):
    def __init__(self, dec_in, d_model, d_ff, n_heads, dropout, attn_dropout=0.):
        super(Channel_Attention_Layer, self).__init__()

        self.att_norm = nn.BatchNorm1d(dec_in)
        self.fft_norm = nn.LayerNorm(d_model)
        
        self.channel_att = _MultiheadAttention(d_model, n_heads, attn_dropout=0., proj_dropout=dropout)
        self.fft = nn.Sequential(
            nn.Linear(d_model, d_ff),
            
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff//2, d_model)
        )
        self.res_attention = False

    def forward(self, q, k, v): 
        if self.res_attention:
            out, attn_weight, scores = self.channel_att(q, k, v)
        else:
            out, attn_weight = self.channel_att(q, k, v)
        out = self.att_norm(out+q)
        out = out + self.fft(out)
        out = self.fft_norm(out)

        return out

class patch_att_block(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, n_layers, n_heads, d_ff, patch_len:int=24, stride:int=24, attn_dropout=0., dropout=0.,
                  padding_patch='end'):
        super(patch_att_block, self).__init__()
        self.pred_pad = False
       
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        
        seq_patch_num = int((seq_len - patch_len)/stride + 2)
        if (pred_len%patch_len) != 0:
            self.pred_pad = True
            pred_patch_num = int((pred_len - patch_len)/stride + 2) 
        else:
            pred_patch_num = pred_len//patch_len
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            
            
        self.W_P = nn.Linear(patch_len, d_model)
        self.PE_i  = nn.Parameter(0.04*torch.rand(seq_patch_num, d_model)-0.02)
        

        self.patch_att = nn.ModuleList([
            Patch_Attention_Layer(d_model, n_heads=n_heads, d_ff=d_ff,
                                                      attn_dropout=attn_dropout, dropout=dropout) for _ in range(n_layers)
        ])

        self.projector = Projection(d_model, patch_len)
        
    def forward(self, inter, inp, pred): 
        
        if self.padding_patch == 'end':
            inp = self.padding_patch_layer(inp)
            if self.pred_pad:
                pred = self.padding_patch_layer(pred)
        inp =  inp.unfold(dimension=-1, size=self.patch_len, step=self.stride)       
        pred = pred.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)   

        inp = self.W_P(inp)+self.PE_i       
        pred = self.W_P(pred)               
        q = torch.reshape(pred, (-1,pred.shape[2],pred.shape[3]))  
        k = torch.cat((inter.unsqueeze(2), inp), dim=2)            
        k = torch.reshape(k, (-1,k.shape[2],k.shape[3]))           
        v = k
        for mod in self.patch_att: q,k,v = mod(q, k, v)
        
        out = self.projector(q) 
        out = torch.reshape(out, (pred.shape[0],pred.shape[1],-1)) 

        return out 

class Patch_Attention_Layer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False, 
                 attn_dropout=0, dropout=0., bias=True, res_attention=False):
        super(Patch_Attention_Layer, self).__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.res_attention = res_attention
        self.cross_attn = _MultiheadAttention(d_model, n_heads, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        
        self.norm_attn = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                GEGLU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff//2, d_model, bias=bias))
        self.norm_ffn = nn.LayerNorm(d_model)
        self.store_attn = store_attn
        self.dropout = nn.Dropout(dropout)

    def forward(self, q:Tensor, k:Tensor, v:Tensor) -> Tensor:
        if self.res_attention:
            pred, attn, scores = self.cross_attn(q, k, v)
        else:
            pred, attn = self.cross_attn(q, k, v)
        pred = q + pred    
        pred = self.norm_attn(pred)
        pred2 = self.ffn(pred)
        pred = pred + pred2   
        pred = self.norm_ffn(pred)
        
        return pred, k, v   


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True):
        
        super().__init__()
        d_h = d_model // n_heads

        self.scale = d_h**-0.5
        self.n_heads, self.d_h = n_heads, d_h

        self.W_Q = nn.Linear(d_model, d_h * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_h * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_h * n_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_h, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q:Tensor, K:Tensor, V:Tensor, prev=None):

        bs = Q.size(0)
        
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_h)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_h) 
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_h) 

        attn_scores = torch.einsum('bphd, bshd -> bphs', q_s, k_s) * self.scale
        
        if prev is not None: attn_scores = attn_scores + prev
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.einsum('bphs, bshd -> bphd', attn_weights, v_s)
        output = output.contiguous().view(bs, -1, self.n_heads*self.d_h)
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class D_kl(nn.Module):
    def __init__(self, feature_dim, z_dim, num_channels, beta):
        super(D_kl, self).__init__()

        self.feature_dim = feature_dim 
        self.z_dim = z_dim
        self.num_channels = num_channels 
        self.beta = beta 
        self.loc_net = nn.ModuleList()
        self.scale_net = nn.ModuleList() 

        for _ in range(num_channels):
            self.loc_net.append(nn.Linear(feature_dim, z_dim))
            
            self.scale_net.append(nn.Sequential(
                nn.Linear(feature_dim, z_dim),
                nn.Softplus() 
            ))
    
    def log_standard_gaussian(self, z):
        log_p = -0.5 * torch.pow(z, 2) - 0.5 * math.log(2 * math.pi)
        return torch.sum(log_p, dim=1) 

    def gaussian_log_prob(self, z, mean, scale):
        scale = scale + 1e-8
        qz_gaussian = torch.distributions.Normal(loc=mean, scale=scale)
        log_q = qz_gaussian.log_prob(z) 
        return torch.sum(log_q, dim=1) 


    def forward(self, x): 
        
        B, C, E = x.shape
        if C != self.num_channels:
            raise ValueError(f"Input channel dimension ({C}) does not match module's num_channels ({self.num_channels})")
        if E != self.feature_dim:
             raise ValueError(f"Input feature dimension ({E}) does not match module's feature_dim ({self.feature_dim})")

        kl_per_channel = [] 

        for i in range(C): 
            
            x_i = x[:, i, :]

            
            mean = self.loc_net[i](x_i)
            
            scale = self.scale_net[i](x_i)

            
            
            scale = scale + 1e-8
            qz_gaussian = torch.distributions.Normal(loc=mean, scale=scale)

            
            qz = qz_gaussian.rsample()

            
            log_p_z = self.log_standard_gaussian(qz)
            
            log_q_z = qz_gaussian.log_prob(qz).sum(dim=1) 

            
            
            
            KL = log_q_z - log_p_z 

            kl_per_channel.append(KL)

        
        stacked_kl = torch.stack(kl_per_channel, dim=0) 

        
        mean_kl = stacked_kl.mean()

        return mean_kl*self.beta

class Norm(nn.Module):
    def __init__(self,channel,output_dim):
        super(Norm, self).__init__()
        self.output_dim=output_dim
    def forward(self, x):        
        self.means = x.mean(1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)      
        x_normalized = (x - self.means) / self.stdev
        return x_normalized
    
    def inverse_normalize(self, x_normalized):
        x_normalized = x_normalized * \
                        (self.stdev[:, 0, :].unsqueeze(1).repeat(
                            1, self.output_dim, 1))
        x_normalized = x_normalized + \
                            (self.means[:, 0, :].unsqueeze(1).repeat(
                                1, self.output_dim, 1)) 
        return x_normalized


class Model(nn.Module):
    def __init__(self, args, **kwargs):
        
        super().__init__()
        self.norm = args.norm 
        self.norm_layer = Norm(args.dec_in, args.pred_len) 

        m_layers = args.m_layers

        
        c_in = args.dec_in
        seq_len = args.seq_len
        self.pred_len = args.pred_len
        n_layers = args.d_layers
        n_heads = args.n_heads
        d_model = args.d_model
        d_ff = args.d_ff
        dropout = args.dropout
        patch_len = args.patch_len
        stride = args.stride
        padding_patch = args.padding_patch
        store_attn = args.store_attn
        m = args.m
        r = args.r
        lambda_ = args.lambda_
        beta = args.beta
        kernel_size = args.kernel_size
        xp_i = args.xp_i

        self.model = Model_backbone(xp_i, kernel_size=kernel_size, m=m, r=r, lambda_=lambda_, beta=beta, m_layers=m_layers, c_in=c_in, seq_len = seq_len, pred_len=self.pred_len, patch_len=patch_len, stride=stride, n_layers=n_layers, 
                                    d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
                                    store_attn=store_attn, padding_patch = padding_patch,  **kwargs)

    
    def forward(self, x):           
        if self.norm:
            x=self.norm_layer(x)        

        x = x.permute(0,2,1)        
        x, ApEn, kl_loss = self.model(x)
        x = x.permute(0,2,1)        

        if self.norm:
            x=self.norm_layer.inverse_normalize(x)
        return x[:,:self.pred_len,:] , ApEn, kl_loss
