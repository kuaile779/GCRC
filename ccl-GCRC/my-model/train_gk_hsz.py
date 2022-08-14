# -*- coding: utf-8 -*-

import torch
import argparse
import time

from data_process import *
from model import gkMRC
from utils import *
from eval_utils import compute_pred
from transformers import AutoModel, AutoConfig,  AutoTokenizer
from transformers import logging
logging.set_verbosity_error()

torch.manual_seed(100)
np.random.seed(100)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epoch", type=int, default=80)
parser.add_argument("--lr", type=float, default=3.0e-5)
parser.add_argument("--max_grad_norm", type=float, default=1)
parser.add_argument("--model_type", type=str, default="./model")
parser.add_argument("--device", default="cuda", type=str, help="Whether not to use CUDA when available")

args = parser.parse_args()

print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print(json.dumps(vars(args), sort_keys=False, indent=2))

with open('./pre_error/error_tensor.json', encoding='utf-8') as f:
    error_tensor = json.load(f)['err_tensor']
error_tensor = torch.FloatTensor(error_tensor)

data = load_file('../data/train-features.obj')
valid_data = load_file('../data/dev-features.obj')
train_data_len = len(data)

batch_size = args.batch_size
config = AutoConfig.from_pretrained(args.model_type)
# tokenizer = AutoTokenizer.from_pretrained(args.model_path)
pretrain_model = AutoModel.from_pretrained(args.model_type, config=config)

model = gkMRC(pretrain_model)
model = model.to(device=args.device)
optimizer = torch.optim.AdamW(model.parameters(),
                              weight_decay=0.01,
                              lr=args.lr)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def get_shuffle_data(data):
    np.random.shuffle(data)
    return data


def iter_printer(total, epoch, type):
    return tqdm(range(0, total, batch_size), desc='epoch {} -- {}'.format(epoch, type))

def train(epoch):
    train_data = get_shuffle_data(data)
    total = len(train_data)
    tr_loss = 0.0
    global_step = 0
    for i in iter_printer(total, epoch, 'train'):
        model.train()
        global_step += 1
        for j in range(i, min(i + batch_size, total)):
            #
            input_ids = [f.input_ids for f in train_data[j]]
            attention_mask = [f.attention_mask for f in train_data[j]]
            token_type_ids = [f.token_type_ids for f in train_data[j]]  #  n * 4 * l
            ans_label = train_data[j][0].ans_label  # 1
            need_error_type = train_data[j][0].need_error_type
            need_evidence = train_data[j][0].need_evidence
            start_evidences = [f.start_evidences for f in train_data[j]]  # n * 4 * l
            end_evidences = [f.end_evidences for f in train_data[j]]
            error_type_label = train_data[j][0].error_type_label  # 4

            #
            input_ids = torch.LongTensor(input_ids).cuda()
            attention_mask = torch.LongTensor(attention_mask).cuda()
            token_type_ids = torch.LongTensor(token_type_ids).cuda()
            ans_label = torch.LongTensor([ans_label]).cuda()
            if need_evidence:
                start_evidences = torch.FloatTensor(start_evidences).cuda()
                end_evidences = torch.FloatTensor(end_evidences).cuda()
            if need_error_type:
                error_type_label = torch.LongTensor(error_type_label).cuda()

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "ans_label": ans_label,
                "need_evidence": need_evidence,
                "need_error_type": need_error_type,
                "start_evidences": start_evidences,
                "end_evidences": end_evidences,
                "error_type_label": error_type_label,
            }

            total_loss = model(**inputs)
            total_loss = total_loss / batch_size
            tr_loss += total_loss.item()
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    print("total_loss:", tr_loss / global_step)


def evaluation(epoch):
    model.eval()
    total = len(valid_data)
    all_results = []

    with torch.no_grad():
        for i in iter_printer(total, epoch, 'eval'):
            for j in range(i, min(i + batch_size, total)):
                #
                input_ids = [f.input_ids for f in valid_data[j]]
                attention_mask = [f.attention_mask for f in valid_data[j]]
                token_type_ids = [f.token_type_ids for f in valid_data[j]]
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
                all_results.append([valid_data[j], answer, start_logits, end_logits, error_type])
    all_f1, all_res = compute_pred(all_results)
    return all_f1, all_res


best_acc = 0.0
for epo in range(args.epoch):
    train(epo)
    accuracy, res = evaluation(epo)
    if accuracy > best_acc:
        best_acc = accuracy
        with open('checkpoint.th', 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)
        with open('./best-dev-pred.json', "w", encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
print('best_acc:', best_acc)