import math
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchmetrics.functional.classification import multilabel_average_precision, multilabel_auroc

import lightning.pytorch as pl

from .drfuse import DrFuseModel
from torcheval.metrics import MultilabelAUPRC

import numpy as np
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from .drfuse import DrFuseModel
import gc
import torchmetrics
import random
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, hamming_loss


# class JSD(nn.Module):
#     def __init__(self):
#         super(JSD, self).__init__()
#         self.kl = nn.KLDivLoss(reduction='none', log_target=True)

#     def forward(self, p: torch.tensor, q: torch.tensor, masks):
#         p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
#         m = (0.5 * (p + q)).log()
#         return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log())).sum() / max(1e-6, masks.sum())
class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.Tensor, q: torch.Tensor, r: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        p, q, r = p.view(-1, p.size(-1)), q.view(-1, q.size(-1)), r.view(-1, r.size(-1))
        
        m_pq = (0.5 * (p + q)).log()
        m_pr = (0.5 * (p + r)).log()
        m_qr = (0.5 * (q + r)).log()

        jsd_pq = 0.5 * (self.kl(m_pq, p.log()) + self.kl(m_pq, q.log()))
        jsd_pr = 0.5 * (self.kl(m_pr, p.log()) + self.kl(m_pr, r.log()))
        jsd_qr = 0.5 * (self.kl(m_qr, q.log()) + self.kl(m_qr, r.log()))

        jsd_total = (jsd_pq + jsd_pr + jsd_qr) / 3.0
        return jsd_total.sum() / max(1e-6, masks.sum())


class DrFuseTrainer(pl.LightningModule):
    def __init__(self, args, label_names):
        super().__init__()
        device = self.device if hasattr(self, "device") else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model = DrFuseModel(hidden_size=args.hidden_size,
                                 num_classes=len(label_names),
                                 ehr_dropout=args.dropout,
                                 ehr_n_head=args.ehr_n_head,
                                 ehr_n_layers=args.ehr_n_layers)
        self.model.to(device)  #  Ensure model is on GPU

        self.save_hyperparameters(args)  # args goes to self.hparams
        print('parameters',self.hparams)
        self.pred_criterion = nn.BCELoss(reduction='none')
        self.alignment_cos_sim = nn.CosineSimilarity(dim=1)
        self.triplet_loss = nn.TripletMarginLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.jsd = JSD()

        # self.val_preds = []
        self.val_preds = {k: [] for k in ['final', 'trend','sea', 'cxr']}
        self.val_labels = []
        self.val_pairs = []

        self.test_preds = []
        self.test_labels = []
        self.test_pairs = []
        self.test_feats = {k: [] for k in ['feat_trend_shared','feat_sea_shared','feat_cxr_shared','feat_trend_distinct',    'feat_sea_distinct','feat_cxr_distinct']}#'feat_ehr_shared', 'feat_ehr_distinct',
                                           #'feat_cxr_shared', 'feat_cxr_distinct']}
        self.test_attns = []

        self.label_names = label_names
        
    def _compute_uncertainty_loss(self, uncertainties, pred_losses):
        # Higher prediction loss should correlate with higher uncertainty
        uncertainty_loss = 0
        modality_mapping = {
            'trend': 'trend',
            'seasonal': 'seasonal',
            'cxr': 'cxr',
            'shared': 'shared'
        }

        for modality, uncertainty_key in modality_mapping.items():
            uncertainty = uncertainties[uncertainty_key]
            pred_loss = pred_losses[modality]
            uncertainty_loss += F.mse_loss(uncertainty, pred_loss.detach())
        return uncertainty_loss

    def _compute_masked_pred_loss(self, input, target, mask):
        return (self.pred_criterion(input, target).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_abs_cos_sim(self, x, y, mask):
        return (self.alignment_cos_sim(x, y).abs() * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_cos_sim(self, x, y, mask):
        return (self.alignment_cos_sim(x, y) * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_mse(self, x, y, mask):
        return (self.mse_loss(x, y).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)

    def _disentangle_loss_jsd(self, model_output, pairs, log=True, mode='train'):
        ehr_mask = torch.ones_like(pairs)
        loss_sim_cxr = self._masked_abs_cos_sim(model_output['feat_cxr_shared'],
                                                model_output['feat_cxr_distinct'], pairs)
        loss_sim_trend = self._masked_abs_cos_sim(model_output['feat_trend_shared'],
                                                model_output['feat_trend_distinct'], ehr_mask)
        
        loss_sim_sea = self._masked_abs_cos_sim(model_output['feat_sea_shared'],
                                                model_output['feat_sea_distinct'], ehr_mask)

        jsd = self.jsd(model_output['feat_trend_shared'].sigmoid(),model_output['feat_sea_shared'].sigmoid(),
                       model_output['feat_cxr_shared'].sigmoid(), pairs)

        loss_disentanglement = (self.hparams.lambda_disentangle_shared * jsd +
                                self.hparams.lambda_disentangle_trend * loss_sim_trend +
                                self.hparams.lambda_disentangle_sea * loss_sim_sea +
                                self.hparams.lambda_disentangle_cxr * loss_sim_cxr)
        if log:
            self.log_dict({
                f'disentangle_{mode}/trend_disinct': loss_sim_trend.detach(),
                f'disentangle_{mode}/sea_disinct': loss_sim_sea.detach(),
                f'disentangle_{mode}/CXR_disinct': loss_sim_cxr.detach(),
                f'disentangle_{mode}/shared_jsd': jsd.detach(),
                'step': float(self.current_epoch)
            }, on_epoch=True, on_step=False, batch_size=pairs.shape[0])

        return loss_disentanglement

    def _compute_prediction_losses(self, model_output, y_gt, pairs, log=True, mode='train'):
        ehr_mask = torch.ones_like(model_output['pred_final'][:, 0])
        loss_pred_final = self._compute_masked_pred_loss(model_output['pred_final'], y_gt, ehr_mask)
        loss_pred_trend = self._compute_masked_pred_loss(model_output['pred_trend'], y_gt, ehr_mask)
        loss_pred_sea = self._compute_masked_pred_loss(model_output['pred_sea'], y_gt, ehr_mask)
        loss_pred_cxr = self._compute_masked_pred_loss(model_output['pred_cxr'], y_gt, pairs)
        loss_pred_shared = self._compute_masked_pred_loss(model_output['pred_shared'], y_gt, ehr_mask)

        if log:
            self.log_dict({
                f'{mode}_loss/pred_final': loss_pred_final.detach(),
                f'{mode}_loss/pred_shared': loss_pred_shared.detach(),
                f'{mode}_loss/pred_trend': loss_pred_trend.detach(),
                f'{mode}_loss/pred_sea': loss_pred_sea.detach(),
                f'{mode}_loss/pred_cxr': loss_pred_cxr.detach(),
                'step': float(self.current_epoch)
            }, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])

        return loss_pred_final, loss_pred_trend,loss_pred_sea, loss_pred_cxr, loss_pred_shared

    def _compute_and_log_loss(self, model_output, y_gt, pairs, log=True, mode='train'):
        prediction_losses = self._compute_prediction_losses(model_output, y_gt, pairs, log, mode)
        loss_pred_final, loss_pred_trend,loss_pred_sea, loss_pred_cxr, loss_pred_shared = prediction_losses

        loss_prediction = (self.hparams.lambda_pred_shared * loss_pred_shared +
                           self.hparams.lambda_pred_ehr * loss_pred_trend +
                           self.hparams.lambda_pred_ehr * loss_pred_sea +
                           self.hparams.lambda_pred_cxr * loss_pred_cxr)

        loss_prediction = loss_pred_final + loss_prediction

        loss_disentanglement = self._disentangle_loss_jsd(model_output, pairs, log, mode)

#         loss_total = loss_prediction + loss_disentanglement
        
        
        epoch_log = {}

        # aux loss for attention ranking
        raw_pred_loss_trend = F.binary_cross_entropy(model_output['pred_trend'].data, y_gt, reduction='none')
        raw_pred_loss_sea = F.binary_cross_entropy(model_output['pred_sea'].data, y_gt, reduction='none')
        raw_pred_loss_cxr = F.binary_cross_entropy(model_output['pred_cxr'].data, y_gt, reduction='none')
        raw_pred_loss_shared = F.binary_cross_entropy(model_output['pred_shared'].data, y_gt, reduction='none')

        pairs = pairs.unsqueeze(1)
        attn_weights = model_output['attn_weights']
#         print(attn_weights.shape)
        attn_trend,attn_sea, attn_shared, attn_cxr = attn_weights[:, :, 0],attn_weights[:, :, 1], attn_weights[:, :, 2], attn_weights[:, :, 3]

        # Compute uncertainty-weighted losses
        weighted_pred_loss_trend = (1 - model_output['uncertainties']['trend']) * raw_pred_loss_trend
        weighted_pred_loss_sea = (1 - model_output['uncertainties']['seasonal']) * raw_pred_loss_sea
        weighted_pred_loss_cxr = (1 - model_output['uncertainties']['cxr']) * raw_pred_loss_cxr
        weighted_pred_loss_shared = (1 - model_output['uncertainties']['shared']) * raw_pred_loss_shared

        # Adjust ranking comparisons using uncertainty-weighted losses
        cxr_overweights_trend = 2 * (weighted_pred_loss_cxr < weighted_pred_loss_trend).float() - 1
        loss_attn1 = pairs * F.margin_ranking_loss(attn_cxr, attn_trend, cxr_overweights_trend, reduction='none')
        loss_attn1 = loss_attn1.sum() / max(1e-6, loss_attn1[loss_attn1 > 0].numel())

        cxr_overweights_sea = 2 * (weighted_pred_loss_cxr < weighted_pred_loss_sea).float() - 1
        loss_attn2 = pairs * F.margin_ranking_loss(attn_cxr, attn_sea, cxr_overweights_sea, reduction='none')
        loss_attn2 = loss_attn2.sum() / max(1e-6, loss_attn2[loss_attn2 > 0].numel())

        trend_overweights_sea = 2 * (weighted_pred_loss_trend < weighted_pred_loss_sea).float() - 1
        loss_attn3 = pairs * F.margin_ranking_loss(attn_trend, attn_sea, trend_overweights_sea, reduction='none')
        loss_attn3 = loss_attn3.sum() / max(1e-6, loss_attn3[loss_attn3 > 0].numel())

        shared_overweights_trend = 2 * (weighted_pred_loss_shared < weighted_pred_loss_trend).float() - 1
        loss_attn4 = pairs * F.margin_ranking_loss(attn_shared, attn_trend, shared_overweights_trend, reduction='none')
        loss_attn4 = loss_attn4.sum() / max(1e-6, loss_attn4[loss_attn4 > 0].numel())

        shared_overweights_sea = 2 * (weighted_pred_loss_shared < weighted_pred_loss_sea).float() - 1
        loss_attn5 = pairs * F.margin_ranking_loss(attn_shared, attn_sea, shared_overweights_sea, reduction='none')
        loss_attn5 = loss_attn5.sum() / max(1e-6, loss_attn5[loss_attn5 > 0].numel())

        shared_overweights_cxr = 2 * (weighted_pred_loss_shared < weighted_pred_loss_cxr).float() - 1
        loss_attn6 = pairs * F.margin_ranking_loss(attn_shared, attn_cxr, shared_overweights_cxr, reduction='none')
        loss_attn6 = loss_attn6.sum() / max(1e-6, loss_attn6[loss_attn6 > 0].numel())

        # Final ranking loss
        loss_attn_ranking = (loss_attn1 + loss_attn2 + loss_attn3 + loss_attn4 + loss_attn5 + loss_attn6) / 6

        
        uncertainty_loss = self._compute_uncertainty_loss(
        model_output['uncertainties'],
            {
                'trend': raw_pred_loss_trend,
                'seasonal': raw_pred_loss_sea,
                'cxr': raw_pred_loss_cxr,
                'shared': raw_pred_loss_shared
            }
        )

        # Combine all auxiliary losses
        auxiliary_losses = (
            self.hparams.lambda_attn_aux * loss_attn_ranking +
            self.hparams.lambda_uncertainty * uncertainty_loss
        )

        # Add to total loss
        loss_total = loss_prediction + loss_disentanglement + auxiliary_losses
        

        if log:
            epoch_log.update({
                f'{mode}_loss/total': loss_total.detach(),
                f'{mode}_loss/prediction': loss_prediction.detach(),
                'step': float(self.current_epoch)
            })
            self.log_dict(epoch_log, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])

        return loss_total

    def _get_batch_data(self, batch):
        x, img, y_ehr, seq_lengths, pairs = batch
        y = torch.from_numpy(y_ehr).float().to(self.device)
        x = torch.from_numpy(x).float().to(self.device)
        img = img.to(self.device)
        pairs = torch.FloatTensor(pairs).to(self.device)
       
        return x, img, y, seq_lengths, pairs

    def _get_alignment_lambda(self):
        if self.hparams.adaptive_adc_lambda:
            lmbda = 2 / (1 + math.exp(-self.hparams.gamma * self.current_epoch)) - 1
        else:
            lmbda = 1
        return lmbda

#     def training_step(self, batch, batch_idx):
#         x, img, y, seq_lengths, pairs = self._get_batch_data(batch)
#         if self.hparams.data_pair == 'paired' and self.hparams.aug_missing_ratio > 0:
#             perm = torch.randperm(pairs.shape[0])
#             idx = perm[:int(self.hparams.aug_missing_ratio * pairs.shape[0])]
#             pairs[idx] = 0
        
#         out = self.model(x, img, seq_lengths, pairs, self._get_alignment_lambda())
#         return self._compute_and_log_loss(out, y_gt=y, pairs=pairs)

#     def training_step(self, batch, batch_idx):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print('device:@@@@@@@@@@@@@@', device)
#         x, img, y, seq_lengths, pairs = self._get_batch_data(batch)

#         # ✅ Move all tensors to GPU
#         x, img, y, seq_lengths, pairs = (
#             x.to(device), img.to(device), y.to(device), seq_lengths.to(device), pairs.to(device)
#         )

#         if self.hparams.data_pair == 'paired' and self.hparams.aug_missing_ratio > 0:
#             perm = torch.randperm(pairs.shape[0], device=device)  # ✅ Ensure permutation is on GPU
#             idx = perm[:int(self.hparams.aug_missing_ratio * pairs.shape[0])]
#             pairs[idx] = 0

#         out = self.model(x, img, seq_lengths, pairs, 0)  # ✅ Model should be on GPU
#         return self._compute_and_log_loss(out, y_gt=y, pairs=pairs)

    def training_step(self, batch, batch_idx):
        device = self.device if hasattr(self, "device") else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.model.to(device)  #  Ensure model is on GPU

        x, img, y, seq_lengths, pairs = self._get_batch_data(batch)
        x, img, y, pairs = x.to(device), img.to(device), y.to(device), pairs.to(device)

        # Ensure all inputs are tensors and moved to GPU
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        img = torch.as_tensor(img, dtype=torch.float32, device=device)
        y = torch.as_tensor(y, dtype=torch.float32, device=device)
        seq_lengths = torch.as_tensor(seq_lengths, dtype=torch.long, device=device)
        pairs = torch.as_tensor(pairs, dtype=torch.float32, device=device)

        if self.hparams.data_pair == 'paired' and self.hparams.aug_missing_ratio > 0:
            perm = torch.randperm(pairs.shape[0], device=device)  #  Ensure permutation is on GPU
            idx = perm[:int(self.hparams.aug_missing_ratio * pairs.shape[0])]
            pairs[idx] = 0  # Ensure modifications happen on GPU

        # Ensure model is also on GPU
        self.model.to(device)

        out = self.model(x, img, seq_lengths, pairs, 0)  # Model is now on GPU
        return self._compute_and_log_loss(out, y_gt=y, pairs=pairs)




    def validation_step(self, batch, batch_idx):
        x, img, y, seq_lengths, pairs = self._get_batch_data(batch)
        out = self.model(x, img, seq_lengths, pairs, self._get_alignment_lambda())
        loss = self._compute_and_log_loss(out, y_gt=y, pairs=pairs, mode='val')
        if self.hparams.attn_fusion == 'avg':
            perd_final = (out['pred_trend']+out['pred_sea'] + out['pred_cxr'] + out['pred_shared']) / 4
            pred_final = ((1 - pairs.unsqueeze(1)) * (out['pred_trend'] + out['pred_shared']+out['pred_sea']) / 3 +
                      pairs.unsqueeze(1) * perd_final)
        else:
            pred_final =  out['pred_final']

        # self.val_preds.append(out['pred_final'])
        self.val_preds['final'].append(pred_final)
        self.val_preds['trend'].append(out['pred_trend'])
        self.val_preds['sea'].append(out['pred_sea'])
        self.val_preds['cxr'].append(out['pred_cxr'])
        self.val_pairs.append(pairs)
        self.val_labels.append(y)

        # return self._compute_masked_pred_loss(out['pred_final'], y, torch.ones_like(y[:, 0]))
        return self._compute_masked_pred_loss(pred_final, y, torch.ones_like(y[:, 0]))

    def on_validation_epoch_end(self):
        for name in ['final', 'trend','sea', 'cxr']:
            y_gt = torch.concat(self.val_labels, dim=0)
            preds = torch.concat(self.val_preds[name], dim=0)
            if name == 'cxr':
                pairs = torch.concat(self.val_pairs, dim=0)
                y_gt = y_gt[pairs==1, :]
                preds = preds[pairs==1, :]

            mlaps = multilabel_average_precision(preds, y_gt.long(), num_labels=y_gt.shape[1], average=None)
            mlroc = multilabel_auroc(preds, y_gt.long(), num_labels=y_gt.shape[1], average=None)

            if name == 'final':
                self.log('Val_PRAUC', mlaps.mean(), logger=False, prog_bar=True)
                self.log('Val_AUROC', mlroc.mean(), logger=False, prog_bar=True)

            log_dict = {
                'step': float(self.current_epoch),
                f'val_PRAUC_avg_over_dxs/{name}': mlaps.mean(),
                f'val_AUROC_avg_over_dxs/{name}': mlroc.mean(),
            }
            for i in range(mlaps.shape[0]):
                log_dict[f'val_PRAUC_per_dx_{name}/{self.label_names[i]}'] = mlaps[i]
                log_dict[f'val_AUROC_per_dx_{name}/{self.label_names[i]}'] = mlroc[i]

            self.log_dict(log_dict)
        print(f'Val==prauc{mlaps.mean()},auroc{mlroc.mean()}')
        for k in self.val_preds:
            self.val_preds[k].clear()
        self.val_pairs.clear()
        self.val_labels.clear()
        
        
        
    def apply_missing_mask(data, missing_percentage):
        """
        Apply a missing mask to the CXR data.

        Args:
            data (torch.Tensor): The CXR data.
            missing_percentage (float): The percentage of data to mask (0 to 1).

        Returns:
            torch.Tensor: The masked CXR data.
        """
        mask = torch.rand_like(data) > missing_percentage
        masked_data = data * mask
        return masked_data
    
    def mask_cxr_images(self,imgs, mask_ratio):
        """
        Completely masks a given percentage of images.

        :param imgs: Tensor of shape (batch_size, C, H, W)
        :param mask_ratio: Percentage of images to mask (e.g., 5 for 5%)
        :return: Modified tensor with masked images
        """
        batch_size = imgs.shape[0]
        num_to_mask = int(batch_size * (mask_ratio / 100))

        if num_to_mask > 0:
            # Select random indices to mask
            mask_indices = random.sample(range(batch_size), num_to_mask)
            # Mask selected images (set them to zero)
            imgs[mask_indices] = torch.zeros_like(imgs[mask_indices])

        return imgs




    def test_step(self, batch, batch_idx, mask_ratio=0):
        """
        Modified test_step that masks a given percentage of CXR images before testing.
        """
        x, img, y, seq_lengths, pairs = self._get_batch_data(batch)

        # Apply masking if mask_ratio > 0
        if mask_ratio > 0:
            img = self.mask_cxr_images(img, mask_ratio)

        out = self.model(x, img, seq_lengths, pairs, 0)
        pred_final = out['pred_final']  # Shape: (batch_size, num_classes)

        # Store predictions and labels for later metric calculation
        self.test_preds.append(pred_final.cpu().detach())  # Store as list of tensors
        self.test_labels.append(y.cpu().detach())          # Store ground truth
        self.test_pairs.append(pairs)
        self.test_attns.append(out['attn_weights'])

        # Return raw outputs for compatibility
        return {"preds": pred_final.cpu().detach(), "labels": y.cpu().detach()}
    


    def on_test_epoch_end(self):
        y_gt = torch.concat(self.test_labels, dim=0)
        preds = torch.concat(self.test_preds, dim=0)
        pairs = torch.concat(self.test_pairs, dim=0)
        attn_weights = torch.concat(self.test_attns, dim=0)

        # Compute AUPRC & AUROC
        mlaps = multilabel_average_precision(preds, y_gt.long(), num_labels=y_gt.shape[1], average=None)
        mlroc = multilabel_auroc(preds, y_gt.long(), num_labels=y_gt.shape[1], average=None)

        # Convert tensors to numpy for sklearn metrics
        binary_preds = (preds.cpu().numpy() > 0.5).astype(int)
        y_true = y_gt.cpu().numpy().astype(int)

        # Compute Micro and Macro F1 scores
        micro_f1 = f1_score(y_true, binary_preds, average="micro")
        macro_f1 = f1_score(y_true, binary_preds, average="macro")

        # Compute Hamming Distance
        hamming_dist = hamming_loss(y_true, binary_preds)

        self.test_results = {
            'y_gt': y_gt.cpu(),
            'preds': preds.cpu(),
            'pairs': pairs.cpu(),
            'mlaps': mlaps.cpu(),
            'mlroc': mlroc.cpu(),
            'prauc': mlaps.mean().item(),
            'auroc': mlroc.mean().item(),
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'hamming_distance': hamming_dist
        }

        for k in self.test_feats:
            self.test_results[k] = torch.concat(self.test_feats[k], dim=0)
            self.test_feats[k].clear()

        self.test_labels.clear()
        self.test_preds.clear()
        self.test_pairs.clear()


    def run_tests_with_masking(self, trainer, test_dl_paired):
        """
        Runs the model on the paired test dataset with different CXR image masking percentages.
        Computes AUPRC, AUROC, F1 Scores, and Hamming Loss.
        """
        mask_levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        results = {}

        for mask_ratio in mask_levels:
            print(f"\nTesting with {mask_ratio}% of CXR images completely masked...\n")

            # Override test_step to apply different mask ratios
            self.test_step = partial(self.test_step, mask_ratio=mask_ratio)

            # Reset storage before each test run
            self.test_preds = []
            self.test_labels = []
            self.test_pairs = []
            self.test_attns = []
            self.test_feats = {}

            # Run test
            trainer.test(model=self, dataloaders=test_dl_paired)

            # Store results after test
            results[f'paired_{mask_ratio}%'] = self.test_results

        return results
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        return optimizer
