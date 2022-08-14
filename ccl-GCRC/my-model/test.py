


# -*- coding: utf-8 -*-
import json
import argparse
import torch
from data_process import *
from model import gkMRC
from utils import *
from transformers import AutoModel, AutoConfig,  AutoTokenizer
from transformers import logging
logging.set_verbosity_error()


def to_list(tensor):
    return tensor.detach().cpu().tolist()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model_type", type=str, default="./model")
parser.add_argument("--device", default="cuda", type=str, help="Whether not to use CUDA when available")
parser.add_argument("--test_data_path", default="../data/test.json", type=str, help="Whether not to use CUDA when available")
parser.add_argument("--pred_path", type=str, default="predictions.json")
parser.add_argument("--pred_model_path", type=str, default="checkpoint.th")

args = parser.parse_args()
model_type = args.model_type
batch_size = args.batch_size

test_data = prepare_test_data(args.test_data_path)

config = AutoConfig.from_pretrained(args.model_type)
# tokenizer = AutoTokenizer.from_pretrained(args.model_path)
pretrain_model = AutoModel.from_pretrained(args.model_type, config=config)

model = gkMRC(pretrain_model)
model.load_state_dict(torch.load(args.pred_model_path, map_location='cpu'))
model.to(device=args.device)

model.eval()
total = len(test_data)
all_results = []

with torch.no_grad():
    for i in tqdm(range(0, total, batch_size)):
        for j in range(i, min(i + batch_size, total)):
            #
            input_ids = [f.input_ids for f in test_data[j]]
            attention_mask = [f.attention_mask for f in test_data[j]]
            token_type_ids = [f.token_type_ids for f in test_data[j]]
            #
            input_ids = torch.LongTensor(input_ids).cuda()
            attention_mask = torch.LongTensor(attention_mask).cuda()
            token_type_ids = torch.LongTensor(token_type_ids).cuda()
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            answer, start_logits, end_logits, error_type = model(**inputs)
            #
            answer = to_list(answer)
            assert len(answer) == 1
            answer = answer[0]
            #
            start_logits = to_list(start_logits)
            end_logits = to_list(end_logits)
            #
            error_type = to_list(error_type)
            all_results.append([test_data[j], answer, start_logits, end_logits, error_type])

all_res = compute_pred_in_test(all_results)

with open(args.pred_path, "w", encoding='utf-8') as f:
    json.dump(all_res, f, indent=2, ensure_ascii=False)

