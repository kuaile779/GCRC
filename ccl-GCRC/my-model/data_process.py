# -*- coding: utf-8 -*-

import json, math
import random
import logging
import collections
from transformers import BasicTokenizer
from dataclasses import dataclass
from typing import List, Dict
from transformers import BertTokenizer
from utils import *
from torch.utils.data import Dataset
from transformers.tokenization_utils import _is_whitespace, _is_punctuation, _is_control
logging.getLogger().setLevel(logging.INFO)

tokenizer = None

@dataclass
class gkExample:
    id: str
    doc_tokens: List
    qa_list: List
    need_evidence: bool
    need_error_type: bool
    start_positions: List = None
    end_positions: List = None
    error_type_label: List = None  # 8分类
    ans_label: int = None

@dataclass
class gkFeature:
    id: str
    unique_id: int
    tokens: List
    doc_tokens: List
    tok_to_orig_map: Dict
    input_ids: List
    attention_mask: List
    token_type_ids: List
    need_evidence: bool
    need_error_type: bool
    start_evidences: List = None
    end_evidences: List = None
    error_type_label: List = None  # 8分类
    ans_label: int = None

SPIECE_UNDERLINE = '▁'
def get_doc_token(context_text):

    def is_whitespace(c):
      if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
        return True
      return False

    def _is_chinese_char(cp):
      if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
      return False

    def is_fuhao(c):
      if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
                or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
                or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
                or c == '‘' or c == '’':
        return True
      return False

    def _tokenize_chinese_chars(text):
      """Adds whitespace around any CJK character."""
      output = []
      for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or is_fuhao(char):
          if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
            output.append(SPIECE_UNDERLINE)
          output.append(char)
          output.append(SPIECE_UNDERLINE)
        else:
          output.append(char)
      return "".join(output)

    context_iter = _tokenize_chinese_chars(context_text)

    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in context_iter:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        if c != SPIECE_UNDERLINE:
            char_to_word_offset.append(len(doc_tokens) - 1)
    return doc_tokens, char_to_word_offset

ans_map = ['A', 'B', 'C', 'D']
error_type_map = ['', 'DTL', 'TEMP', 'SPOT', 'NAS', 'ITQ', 'CAUS', 'ITA']
error_type_chinese_map = ['无错误', '细节错误', '时间属性错误', '主谓不一致', '充要条件错误', '答非所问', '因果错误', '无中生有']
def read_examples(data, is_training):

    def get_new_evioffset(e, passage):
        sent_end_char = ['。', '.', '！', '!', '？', '?']
        index_sum = []
        for i in range(len(passage)-len(e)+1):
            temp_pass = passage[i: i+len(e)]
            common = collections.Counter(e) & collections.Counter(temp_pass)
            num_same = sum(common.values())
            index_sum.append(num_same)
        start =  index_sum.index(max(index_sum))
        end = start + len(e) - 1
        if passage[end] in sent_end_char:
            new_end = end
        else:
            aft_e_tokens = passage[end + 1:]
            all_e = []
            for c in sent_end_char:
                if c in aft_e_tokens:
                    temp_e = end + 1 + aft_e_tokens.index(c)
                    all_e.append(temp_e)
            if len(all_e) > 0:
                new_end = min(all_e)
            else:
                new_end = len(passage) - 1
        # if new_end - start + 1 > 500:
        #     print(1)
        #     new_end = end
        return start, new_end

    examples = []
    for pre_data in tqdm(data):
        id = pre_data['id']
        passage = pre_data['passage']
        question = pre_data['question']
        options = pre_data['options']
        assert len(options) == 4
        qa_list = []
        for j in range(4):
            option = options[j]
            qa_cat = " ".join([question, option])
            qa_list.append(qa_cat)
        # print(qa_list)

        # doc_token
        doc_tokens, char_to_word_offset = get_doc_token(passage)

        ans_label = None  # 1
        need_evidence = None
        need_error_type = None
        start_positions = None  # [[],[],[],[]]
        end_positions = None
        error_type_label = None  # [1,2,3,4]
        if is_training:
            answer = pre_data['answer']
            ans_label = ans_map.index(answer)
            evidences = pre_data['evidences']
            error_type = pre_data['error_type']
            if evidences == []:
                need_evidence = False
            else:
                need_evidence = True
                start_positions = []
                end_positions = []
                for j in range(4):
                    evi = evidences[j]
                    temp_start_positions = []
                    temp_end_positions = []
                    for e in evi:
                        if e not in passage:
                            evi_offset, evi_end = get_new_evioffset(e, passage)
                        else:
                            evi_offset = passage.index(e)
                            evi_end = evi_offset + len(e) - 1
                        sp = char_to_word_offset[evi_offset]
                        ep =  char_to_word_offset[evi_end]
                        assert sp <= ep
                        temp_start_positions.append(sp)
                        temp_end_positions.append(ep)
                    start_positions.append(temp_start_positions)
                    end_positions.append(temp_end_positions)

            if len(error_type) == 0:
                need_error_type = False
            else:
                need_error_type = True
                error_type_label = []
                for et in error_type:
                    error_type_label.append(error_type_map.index(et))

        examples.append(gkExample(
            id=id,
            doc_tokens=doc_tokens,
            qa_list=qa_list,
            need_evidence=need_evidence,
            need_error_type=need_error_type,
            start_positions=start_positions,
            end_positions=end_positions,
            error_type_label=error_type_label,
            ans_label=ans_label
        ))

    return examples

def convert_single_example_to_features(example:gkExample, is_training, max_seq_length=512,
                                       max_qa_length=104, doc_stride=128):

    def get_split_text(text, split_len, overlap_len):
        split_text = []
        window = split_len - overlap_len
        w = 0
        while w * window + split_len < len(text):
            text_piece = text[w * window: w * window + split_len]
            w += 1
            split_text.append(text_piece)

        if text[w * window:]:
            split_text.append(text[w * window:])
        else:
            print("1111111")
        return split_text

    # error_type_class = [y for x in error_type_chinese_map for y in [1] + tokenizer.encode(x, add_special_tokens=False)]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):  # 将文章变为bert分词后的文章
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    # 截断过长qa对
    for j in range(4):
        temp = tokenizer.tokenize(example.qa_list[j])
        if len(temp) > max_qa_length:
            temp = temp[- max_qa_length:]
        example.qa_list[j] = temp

    temp_max_qa_len = 0
    for j in range(4):
        qa_tokens = example.qa_list[j]
        if len(qa_tokens) > temp_max_qa_len:
            temp_max_qa_len = len(qa_tokens)

    # split_len = max_seq_length - temp_max_qa_len - len(error_type_class) - 4
    split_len = max_seq_length - temp_max_qa_len - 3
    p_span = get_split_text(all_doc_tokens, split_len, doc_stride)

    # global count
    # if str(len(p_span)) not in count:
    #     count[str(len(p_span))] = 1
    # else:
    #     count[str(len(p_span))] += 1

    if len(p_span)>6:
        p_span = p_span[:6]

    exam2features = []
    for p in p_span:
        temp_tok_to_orig_map = {}

        p_start = None
        for k in range(len(all_doc_tokens)):
            if all_doc_tokens[k:k+len(p)] == p:
                p_start = k
        assert p_start != None
        p_end = p_start + len(p) - 1

        for k in range(len(p)):
            temp_tok_to_orig_map[k+1] = tok_to_orig_index[p_start+k]  # cls

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []

        start_evidences = []  #  4 * l
        end_evidences = []
        for k in range(4):
            qa_tokens = example.qa_list[k]
            qa_ids = tokenizer.convert_tokens_to_ids(qa_tokens)
            second_tokens = qa_ids
            # print(second_tokens)

            encoded_dict = tokenizer.encode_plus(
                p,
                second_tokens,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                padding="max_length",
                truncation="only_second",
                return_token_type_ids=True
            )
            assert encoded_dict['overflowing_tokens'] == []

            input_ids = encoded_dict['input_ids']
            attention_mask = encoded_dict['attention_mask']
            token_type_ids = encoded_dict['token_type_ids']
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)

            # evi
            if example.need_evidence:
                temp_start_evidences = [0] * max_seq_length
                temp_end_evidences = [0] * max_seq_length
                sp = example.start_positions[k]
                ep = example.end_positions[k]
                for l in range(len(sp)):
                    s = orig_to_tok_index[sp[l]]
                    e = orig_to_tok_index[ep[l]]
                    if s>=p_start and e<=p_end :
                        temp_start_evidences[1+s-p_start] = 1
                        temp_end_evidences[1+e-p_start] = 1
                if 1 not in temp_start_evidences:
                    temp_start_evidences[0] = 1
                    temp_end_evidences[0] = 1
                start_evidences.append(temp_start_evidences)
                end_evidences.append(temp_end_evidences)


        exam2features.append(gkFeature(
            id='111',
            unique_id=0,
            tokens=p,
            doc_tokens=example.doc_tokens,
            tok_to_orig_map=temp_tok_to_orig_map,
            input_ids=all_input_ids,
            attention_mask=all_attention_mask,
            token_type_ids=all_token_type_ids,
            need_evidence=example.need_evidence,
            need_error_type=example.need_error_type,
            start_evidences=start_evidences,
            end_evidences=end_evidences,
            error_type_label=example.error_type_label,
            ans_label=example.ans_label
        ))

    return exam2features


def convert_to_features(filename, is_training):

    with open(filename, encoding='utf-8') as f:
        raw = json.load(f)
        data = raw['data']
        examples = read_examples(data, is_training)
        logging.info('get {} examples'.format(len(examples)))
        feature_list = []
        unique_id = 0
        for example in tqdm(examples):
            feature = convert_single_example_to_features(example, is_training)
            for fe in feature:
                fe.id = example.id
                fe.unique_id = unique_id
                unique_id += 1
            feature_list.append(feature)
    logging.info('get {} features'.format(len(feature_list)))
    return examples, feature_list


def prepare_bert_data(model_type='./model'):
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_type)
    print("--------------")
    if not os.path.exists('../data/train-features.obj'):
        examples, features = convert_to_features('../data/train.json', is_training=True)
        # dump_file(examples, '../data/train-examples.obj')
    #     dump_file(features, '../data/train-features.obj')
    #
    # print("--------------")
    # if not os.path.exists('../data/dev-features.obj'):
    #     examples, features = convert_to_features('../data/valid.json', is_training=False)
    #     # dump_file(examples, '../data/dev-examples.obj')
    #     dump_file(features, '../data/dev-features.obj')
    #
    # print("--------------")
    # if not os.path.exists('../data/test-features.obj'):
    #     examples, features = convert_to_features('../data/test.json', is_training=False)
    #     # dump_file(examples, '../data/test-examples.obj')
    #     dump_file(features, '../data/test-features.obj')


def prepare_test_data(test_path, model_type='./model'):
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_type)
    print("--------------")
    examples, features = convert_to_features(test_path, is_training=False)
    return features

count = {}
prepare_bert_data()  # 6994/16803 863/1993 10000/24754  # 26588 3192 38818
# prepare_test_data('../data/test.json')
print(count)

