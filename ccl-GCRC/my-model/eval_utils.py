from data_process import *
from eval import eval_acc

@dataclass
class gkPrediction:
    feature: gkFeature
    start_index: int
    end_index: int
    start_logit: float
    end_logit: float
    text: str = None
    orig_start_index: int = None
    orig_end_index: int = None
    final_score: int = None

tokenizer = None
def compute_pred(all_results, n_best_size=20, max_answer_length=250, model_type='./model', multi_span_threshold=0.8):
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_type)

    def _get_best_indexes(logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes


    all_pred = {'data':[]}
    for res in all_results:
        features = res[0] # n
        id = features[0].id
        answer = res[1]
        start_logits = res[2]  #  n * 4 * l
        end_logits = res[3]
        error_type = res[4]

        # ans
        answer = ans_map[answer]
        # err
        error_res = [error_type_map[e.index(max(e))] for e in error_type]
        assert len(error_res) == 4

        # evi
        evi_res = []
        doc_tokens = features[0].doc_tokens
        for i in range(4):
            predictions = []
            for j in range(len(start_logits)):
                feature = features[j]
                se = start_logits[j][i]  # l
                ee = end_logits[j][i]
                start_indexes = _get_best_indexes(se, n_best_size)
                end_indexes = _get_best_indexes(ee, n_best_size)
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if start_index >= len(feature.tokens)+1:
                            continue
                        if end_index >= len(feature.tokens)+1:
                            continue
                        if start_index not in feature.tok_to_orig_map:
                            continue
                        if end_index not in feature.tok_to_orig_map:
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        predictions.append(
                            gkPrediction(
                                feature=feature,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=se[start_index],
                                end_logit=ee[end_index],
                            )
                        )

            # 无效证据
            if predictions == []:
                evi_res.append([""])
                continue

            predictions = sorted(predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
            seen_predictions = {}
            filtered_predictions = []

            for prediction in predictions:
                if len(filtered_predictions) >= n_best_size:
                    break
                feature = prediction.feature
                tok_tokens = feature.tokens[prediction.start_index-1: prediction.end_index]
                orig_doc_start = feature.tok_to_orig_map[prediction.start_index]
                orig_doc_end = feature.tok_to_orig_map[prediction.end_index]
                orig_tokens = doc_tokens[orig_doc_start: (orig_doc_end + 1)]
                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = "".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text)
                prediction.orig_start_index = orig_doc_start
                prediction.orig_end_index = orig_doc_end
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
                prediction.text = final_text
                filtered_predictions.append(prediction)

            predictions = filtered_predictions
            score_normalization(predictions)
            best_non_null_entry = None
            for p in predictions:
                if best_non_null_entry is None:
                    best_non_null_entry = p
                    break
            max_score = best_non_null_entry.final_score
            span_covered = [0 for i in range(len(doc_tokens))]
            predict_answers = []
            for p in predictions:
                if p.final_score > (max_score * multi_span_threshold) \
                        and 1 not in span_covered[p.orig_start_index: (p.orig_end_index + 1)]:
                    predict_answers.append(p.text)
                    span_covered[p.orig_start_index: (p.orig_end_index + 1)] = [1 for i in range(p.orig_start_index,
                                                                                                 p.orig_end_index + 1)]
            evi_res.append(predict_answers)

        all_pred['data'].append({
            'id': id,
            'answer': answer,
            'evidences': evi_res,
            'error_type': error_res,
        })

    with open('./pred-dev.json', "w", encoding='utf-8') as f:
        json.dump(all_pred, f, indent=2, ensure_ascii=False)
    score_avg = eval_acc('pred-dev.json', '../data/valid.json')
    return score_avg, all_pred


def get_final_text(pred_text, orig_text, do_lower_case=False, verbose_logging=False):

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logging.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logging.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position: (orig_end_position + 1)]
    return output_text


def score_normalization(predictions: List[gkPrediction]):
    scores = [p.start_logit + p.end_logit for p in predictions]
    max_score = max(scores)
    min_score = min(scores)
    for p in predictions:
        if (max_score - min_score) == 0:
            p.final_score = 0
            continue
        p.final_score = 1.0 * ((p.start_logit + p.end_logit) - min_score) / (max_score - min_score)

def compute_pred_in_test(all_results, n_best_size=20, max_answer_length=250, model_type='./model',
                 multi_span_threshold=0.8):
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_type)

    def _get_best_indexes(logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    all_pred = {'data': []}
    for res in all_results:
        features = res[0]  # n
        id = features[0].id
        answer = res[1]
        start_logits = res[2]  # n * 4 * l
        end_logits = res[3]
        error_type = res[4]

        # ans
        answer = ans_map[answer]
        # err
        error_res = [error_type_map[e.index(max(e))] for e in error_type]
        assert len(error_res) == 4

        # evi
        evi_res = []
        doc_tokens = features[0].doc_tokens
        for i in range(4):
            predictions = []
            for j in range(len(start_logits)):
                feature = features[j]
                se = start_logits[j][i]  # l
                ee = end_logits[j][i]
                start_indexes = _get_best_indexes(se, n_best_size)
                end_indexes = _get_best_indexes(ee, n_best_size)
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if start_index >= len(feature.tokens)+1:
                            continue
                        if end_index >= len(feature.tokens)+1:
                            continue
                        if start_index not in feature.tok_to_orig_map:
                            continue
                        if end_index not in feature.tok_to_orig_map:
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        predictions.append(
                            gkPrediction(
                                feature=feature,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=se[start_index],
                                end_logit=ee[end_index],
                            )
                        )

            # 无效证据
            if predictions == []:
                evi_res.append([""])
                continue

            predictions = sorted(predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
            seen_predictions = {}
            filtered_predictions = []

            for prediction in predictions:
                if len(filtered_predictions) >= n_best_size:
                    break
                feature = prediction.feature
                tok_tokens = feature.tokens[prediction.start_index - 1: prediction.end_index]
                orig_doc_start = feature.tok_to_orig_map[prediction.start_index]
                orig_doc_end = feature.tok_to_orig_map[prediction.end_index]
                orig_tokens = doc_tokens[orig_doc_start: (orig_doc_end + 1)]
                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = "".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text)
                prediction.orig_start_index = orig_doc_start
                prediction.orig_end_index = orig_doc_end
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
                prediction.text = final_text
                filtered_predictions.append(prediction)

            predictions = filtered_predictions
            score_normalization(predictions)
            best_non_null_entry = None
            for p in predictions:
                if best_non_null_entry is None:
                    best_non_null_entry = p
                    break
            max_score = best_non_null_entry.final_score
            span_covered = [0 for i in range(len(doc_tokens))]
            predict_answers = []
            for p in predictions:
                if p.final_score > (max_score * multi_span_threshold) \
                        and 1 not in span_covered[p.orig_start_index: (p.orig_end_index + 1)]:
                    predict_answers.append(p.text)
                    span_covered[p.orig_start_index: (p.orig_end_index + 1)] = [1 for i in
                                                                                range(p.orig_start_index,
                                                                                      p.orig_end_index + 1)]
            evi_res.append(predict_answers)

        all_pred['data'].append({
            'id': id,
            'answer': answer,
            'evidences': evi_res,
            'error_type': error_res,
        })

    return all_pred