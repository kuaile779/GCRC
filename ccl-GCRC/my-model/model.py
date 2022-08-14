# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCELoss

class gkMRC(nn.Module):
    def __init__(self, pretrain_model):
        super().__init__()
        self.encoder = pretrain_model
        self.n_hidden = self.encoder.config.hidden_size

        self.ans_prediction = nn.Linear(self.n_hidden, 1)
        self.evi_prediction = nn.Linear(self.n_hidden, 2)
        self.error_prediction = nn.Linear(self.n_hidden, 8)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                ans_label=None,
                need_evidence=None,
                need_error_type=None,
                start_evidences=None,
                end_evidences=None,
                error_type_label=None,
            ):

        #
        input_ids = input_ids.view(-1, input_ids.size(2))  # (n * 4) * l
        attention_mask = attention_mask.view(-1, attention_mask.size(2))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(2))

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs[0]  # (n*4) * l * h

        # ans
        ans_hidden = hidden.select(1, 0)  # (n*4) * h
        ans_hidden = ans_hidden.view(-1, 4, self.n_hidden)  # n * 4 * h
        ans_pred = self.ans_prediction(ans_hidden).squeeze(-1)  # n * 4
        ans_pred = torch.max(ans_pred, dim=0)[0].unsqueeze(0)  # 1 * 4

        # evi
        evi_pred = self.evi_prediction(hidden)  # (n*4) * l * 2
        start_logits, end_logits = evi_pred.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (n*4) * l
        start_logits = start_logits.view(-1, 4, start_logits.size(1))  # n*4*l
        end_logits = end_logits.squeeze(-1)  # (n*4) * l
        end_logits = end_logits.view(-1, 4, end_logits.size(1))  # n*4*l

        # error
        mask_idx = torch.eq(input_ids, 102)  # 1 is the index in the seq we separate each candidates.
        err_hidden = hidden.masked_select(mask_idx.unsqueeze(2).expand_as(hidden)).view(
            hidden.size(0), -1, self.n_hidden)  # (n*4) * 2 * h
        err_hidden = err_hidden.select(1, 1)  # (n*4) * h
        err_hidden = err_hidden.view(-1, 4, self.n_hidden)  # n * 4 * h
        err_pred = self.error_prediction(err_hidden) # n * 4 * 8
        err_pred = torch.max(err_pred, dim=0)[0]  # 4 * 8

        if ans_label is None:
            return ans_pred.argmax(1), start_logits, end_logits, err_pred
        else:
            ans_loss = F.cross_entropy(ans_pred, ans_label)

            evi_loss = 0
            if need_evidence:
                loss_fct = BCELoss(reduction="mean")
                start_loss = loss_fct(torch.sigmoid(start_logits), start_evidences)
                end_loss = loss_fct(torch.sigmoid(end_logits), end_evidences)
                evi_loss = (start_loss + end_loss) / 2

            err_loss = 0
            if need_error_type:
                err_loss = F.cross_entropy(err_pred, error_type_label)

            total_loss = ans_loss + 3 * (evi_loss + err_loss)
            return total_loss
