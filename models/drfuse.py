import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights

from .ehr_transformer import EHRTransformer


class UncertaintyAwareModalFusion(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Attention projection layer (similar to original)
        self.attn_proj = nn.Linear(hidden_size, (2+num_classes)*hidden_size)
        self.final_pred_fc = nn.Linear(hidden_size, num_classes)

        # Uncertainty networks remain the same
        self.uncertainty_nets = nn.ModuleDict({
            'trend': nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, 1),
                nn.Sigmoid()
            ).to(self.device),
            'seasonal': nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, 1),
                nn.Sigmoid()
            ).to(self.device),
            'cxr': nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, 1),
                nn.Sigmoid()
            ).to(self.device),
            'shared': nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, 1),
                nn.Sigmoid()
            ).to(self.device)
        })

    def forward(self, feat_trend_distinct, feat_sea_distinct, feat_cxr_distinct, feat_shared, pairs):
        # 1. Get uncertainties
        uncertainties = {
            'trend': self.uncertainty_nets['trend'](feat_trend_distinct),
            'seasonal': self.uncertainty_nets['seasonal'](feat_sea_distinct),
            'cxr': self.uncertainty_nets['cxr'](feat_cxr_distinct),
            'shared': self.uncertainty_nets['shared'](feat_shared)
        }

        # 2. Disease-wise Attention with uncertainty
        attn_input = torch.stack([
            feat_trend_distinct,
            feat_sea_distinct,
            feat_shared,
            feat_cxr_distinct
        ], dim=1)

        # Project for attention
        qkvs = self.attn_proj(attn_input)
        q, v, *k = qkvs.chunk(2+self.num_classes, dim=-1)

        # Compute uncertainty-weighted query vector
        uncertainty_weights = torch.stack([
            1 - uncertainties['trend'],
            1 - uncertainties['seasonal'],
            1 - uncertainties['shared'],
            1 - uncertainties['cxr']
        ], dim=1)

        # Weighted mean query incorporating uncertainties
        q_weighted = q * uncertainty_weights
        q_mean = pairs * q_weighted.mean(dim=1) + (1-pairs) * q_weighted[:, :-1].mean(dim=1)

        # Compute attention weights
        ks = torch.stack(k, dim=1)
        attn_logits = torch.einsum('bd,bnkd->bnk', q_mean, ks)
        attn_logits = attn_logits / math.sqrt(q.shape[-1])

        # Apply masking for unpaired samples
        attn_mask = torch.ones_like(attn_logits)
        attn_mask[pairs.squeeze()==0, :, -1] = 0
        attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))

        # Get uncertainty-aware attention weights
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Final fusion with uncertainty-weighted attention
        feat_final = torch.matmul(attn_weights, v)

        # Get predictions
        pred_logits = self.final_pred_fc(feat_final)
        pred_final = torch.diagonal(pred_logits, dim1=1, dim2=2).sigmoid()

        return feat_final, uncertainties, attn_weights
    
class DrFuseModel(nn.Module):
    def __init__(self, hidden_size, num_classes, ehr_dropout, ehr_n_layers, ehr_n_head,
           cxr_model='swin_s', logit_average=False):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size=hidden_size
        self.logit_average = logit_average
        self.ehr_model = EHRTransformer(input_size=76, num_classes=num_classes,
                                      d_model=hidden_size, n_head=ehr_n_head,
                                      n_layers_feat=1, n_layers_shared=ehr_n_layers,
                                      n_layers_distinct=ehr_n_layers,
                                      dropout=ehr_dropout)

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cxr_model_feat = nn.Sequential(
          resnet.conv1,
          resnet.bn1,
          resnet.relu,
          resnet.maxpool,
        )

        self.cxr_model_shared = nn.Sequential(
          resnet.layer1,
          resnet.layer2,
          resnet.layer3,
          resnet.layer4,
          nn.AdaptiveAvgPool2d((1, 1)),
          nn.Flatten(),
          nn.Linear(2048, hidden_size)
        )

        self.cxr_model_spec = nn.Sequential(
          resnet.layer1,
          resnet.layer2,
          resnet.layer3,
          resnet.layer4,
          nn.AdaptiveAvgPool2d((1, 1)),
          nn.Flatten(),
          nn.Linear(2048, hidden_size)
        )

        self.shared_project = nn.Sequential(
          nn.Linear(hidden_size, hidden_size*2),
          nn.ReLU(),
          nn.Linear(hidden_size*2, hidden_size),
          nn.ReLU(),
          nn.Linear(hidden_size, hidden_size)
        )

        self.ehr_model_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.cxr_model_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.fuse_model_shared = nn.Linear(in_features=hidden_size, out_features=num_classes)

        self.domain_classifier = nn.Sequential(
          nn.Linear(hidden_size, hidden_size//2),
          nn.ReLU(),
          nn.Linear(hidden_size//2, 1)
        )
        self.attn_proj = nn.Linear(hidden_size, (2+num_classes)*hidden_size)
        self.final_pred_fc = nn.Linear(hidden_size, num_classes)
        self.fusion_module = UncertaintyAwareModalFusion(hidden_size, num_classes)
        
        
    def forward(self, x, img, seq_lengths, pairs, grl_lambda):
        # Extract features
        feat_trend_shared, feat_trend_distinct, feat_sea_shared, feat_sea_distinct, tpred_ehr, spred_ehr = self.ehr_model(x, seq_lengths)
        feat_cxr = self.cxr_model_feat(img)
        feat_cxr_shared = self.cxr_model_shared(feat_cxr)
        feat_cxr_distinct = self.cxr_model_spec(feat_cxr)

        # Initial predictions
        pred_cxr = self.cxr_model_linear(feat_cxr_distinct).sigmoid()

        # Project shared features
        feat_trend_shared = self.shared_project(feat_trend_shared)
        feat_sea_shared = self.shared_project(feat_sea_shared)
        feat_cxr_shared = self.shared_project(feat_cxr_shared)

        pairs = pairs.unsqueeze(1)
        h0=feat_trend_shared
        h1 = feat_sea_shared
        h2 = feat_cxr_shared
        term1 = torch.stack([h0+h1+h2, h0+h1+h2, h0+h1+h2, h0, h1, h2], dim=2)
        term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1),torch.zeros_like(h1), h0, h1, h2], dim=2)
        feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)
        
        
        # Apply logit pooling when CXR is missing
        feat_logit_pooling = torch.log((2 * torch.exp(h0 + h1) + torch.exp(h0) + torch.exp(h1)) / (2 + torch.exp(h0) + torch.exp(h1)))

        # Select the appropriate shared feature based on CXR availability
        feat_avg_shared = pairs * feat_avg_shared + (1 - pairs) * feat_logit_pooling

        # Final shared prediction
        pred_shared = self.fuse_model_shared(feat_avg_shared).sigmoid()

        
        
##################original
#         feat_avg_shared = pairs * feat_avg_shared + (1 - pairs) * (h0+h1)/2
#         pred_shared = self.fuse_model_shared(feat_avg_shared).sigmoid()


        # Uncertainty-aware fusion
#         fusion_module = UncertaintyAwareModalFusion(self.hidden_size, self.num_classes)
        feat_final, uncertainties, attn_weights = self.fusion_module(
            feat_trend_distinct,
            feat_sea_distinct,
            feat_cxr_distinct,
            feat_avg_shared,
            pairs
        )

        # Final predictions
        pred_final = self.final_pred_fc(feat_final)
        pred_final = torch.diagonal(pred_final, dim1=1, dim2=2).sigmoid()

        outputs = {
            'feat_trend_shared': feat_trend_shared,
            'feat_sea_shared': feat_sea_shared,
            'feat_cxr_shared': feat_cxr_shared,
            'feat_trend_distinct': feat_trend_distinct,
            'feat_sea_distinct': feat_sea_distinct,
            'feat_cxr_distinct': feat_cxr_distinct,
            'feat_final': feat_final,
            'pred_final': pred_final,
            'pred_shared': pred_shared,
            'pred_trend': tpred_ehr,
            'pred_sea': spred_ehr,
            'pred_cxr': pred_cxr,
            'attn_weights': attn_weights,
            'uncertainties': uncertainties
        }
        return outputs

