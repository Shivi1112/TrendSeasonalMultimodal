import torch
from torch import nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.rand(1, max_len, d_model))
        self.pe.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # x: (batch_size, seq_len, embedding_dim)
        return self.dropout(x)


import torch
import numpy as np
from torch import nn
from statsmodels.tsa.seasonal import STL

class EHRTransformer(nn.Module):
    def __init__(self, input_size, num_classes,
                 d_model=256, n_head=8, n_layers_feat=1,
                 n_layers_shared=1, n_layers_distinct=1,
                 dropout=0.3, max_len=350):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.emb = nn.Linear(input_size, d_model)
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=0, max_len=max_len)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_feat = nn.TransformerEncoder(layer, num_layers=n_layers_feat)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_shared = nn.TransformerEncoder(layer, num_layers=n_layers_shared)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_distinct = nn.TransformerEncoder(layer, num_layers=n_layers_distinct)
        self.fc_distinct = nn.Linear(d_model, num_classes)

    def forward(self, x, seq_lengths):
        attn_mask = torch.stack([torch.cat([torch.zeros(len_, device=x.device),
                                 float('-inf')*torch.ones(max(seq_lengths)-len_, device=x.device)])
                                for len_ in seq_lengths])
        trend, seasonal = self.extract_trend_seasonal(x)
        
        # Ensure the same dtype for all tensors before passing through the Linear layer
        trend = trend.type_as(x)
        seasonal = seasonal.type_as(x)
        
        trend = self.emb(trend) # * math.sqrt(self.d_model)
        trend = self.pos_encoder(trend)
        seasonal = self.emb(seasonal) # * math.sqrt(self.d_model)
        seasonal = self.pos_encoder(seasonal)
        
        t_feat = self.model_feat(trend, src_key_padding_mask=attn_mask)
        s_feat = self.model_feat(seasonal, src_key_padding_mask=attn_mask)
        
        ht_shared = self.model_shared(t_feat, src_key_padding_mask=attn_mask)
        ht_distinct = self.model_distinct(t_feat, src_key_padding_mask=attn_mask)

        hs_shared = self.model_shared(s_feat, src_key_padding_mask=attn_mask)
        hs_distinct = self.model_distinct(s_feat, src_key_padding_mask=attn_mask)
        
        padding_mask = torch.ones_like(attn_mask).unsqueeze(2)
        padding_mask[attn_mask == float('-inf')] = 0
        trep_shared = (padding_mask * ht_shared).sum(dim=1) / padding_mask.sum(dim=1)
        trep_distinct = (padding_mask * ht_distinct).sum(dim=1) / padding_mask.sum(dim=1)
        
        srep_shared = (padding_mask * hs_shared).sum(dim=1) / padding_mask.sum(dim=1)
        srep_distinct = (padding_mask * hs_distinct).sum(dim=1) / padding_mask.sum(dim=1)

        tpred_distinct = self.fc_distinct(trep_distinct).sigmoid()
        spred_distinct = self.fc_distinct(srep_distinct).sigmoid()
        
        return trep_shared, trep_distinct, srep_shared, srep_distinct, tpred_distinct, spred_distinct

    def extract_trend_seasonal(self, data, period=12):
        trend = []
        seasonal = []
        values = data.cpu().numpy()  # Move tensor to CPU and convert to NumPy array
        batch_size, time_steps, num_variables = values.shape
        
        for b in range(batch_size):
            trend_batch = []
            seasonal_batch = []
            for var in range(num_variables):
                series = values[b, :, var]
                stl = STL(series, period=period)
                result = stl.fit()
                trend_batch.append(result.trend)
                seasonal_batch.append(result.seasonal)
            trend.append(np.stack(trend_batch, axis=1))
            seasonal.append(np.stack(seasonal_batch, axis=1))

        trend = np.stack(trend, axis=0)
        seasonal = np.stack(seasonal, axis=0)

        trend = torch.tensor(trend, device=data.device, dtype=data.dtype)
        seasonal = torch.tensor(seasonal, device=data.device, dtype=data.dtype)

        return trend, seasonal
