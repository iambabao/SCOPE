# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/1/1
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/11/29
"""

import json
import random
import logging
import numpy as np

logger = logging.getLogger(__name__)


def init_logger(level, filename=None, mode='a', encoding='utf-8'):
    logging_config = {
        'format': '%(asctime)s - %(levelname)s - %(name)s:\t%(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
        'level': level,
        'handlers': [logging.StreamHandler()]
    }
    if filename:
        logging_config['handlers'].append(logging.FileHandler(filename, mode, encoding))
    logging.basicConfig(**logging_config)


def log_title(title, sep='='):
    return sep * 50 + '  {}  '.format(title) + sep * 50


def read_file(filename, mode='r', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fin:
        for line in fin:
            if skip > 0:
                skip -= 1
                continue
            yield line


def save_file(data, filename, mode='w', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fout:
        for line in data:
            if skip > 0:
                skip -= 1
                continue
            print(line, file=fout)


def read_json(filename, mode='r', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fin:
        return json.load(fin)


def save_json(data, filename, mode='w', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)


def read_json_lines(filename, mode='r', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fin:
        for line in fin:
            if skip > 0:
                skip -= 1
                continue
            yield json.loads(line)


def save_json_lines(data, filename, mode='w', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fout:
        for line in data:
            if skip > 0:
                skip -= 1
                continue
            print(json.dumps(line, ensure_ascii=False), file=fout)


def read_txt_dict(filename, sep=None, mode='r', encoding='utf-8', skip=0):
    key_2_id = dict()
    with open(filename, mode, encoding=encoding) as fin:
        for line in fin:
            if skip > 0:
                skip -= 1
                continue
            key, _id = line.strip().split(sep)
            key_2_id[key] = _id
    id_2_key = dict(zip(key_2_id.values(), key_2_id.keys()))

    return key_2_id, id_2_key


def save_txt_dict(key_2_id, filename, sep=None, mode='w', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fout:
        for key, value in key_2_id.items():
            if skip > 0:
                skip -= 1
                continue
            if sep:
                print('{} {}'.format(key, value), file=fout)
            else:
                print('{}{}{}'.format(key, sep, value), file=fout)


def read_json_dict(filename, mode='r', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fin:
        key_2_id = json.load(fin)
        id_2_key = dict(zip(key_2_id.values(), key_2_id.keys()))

    return key_2_id, id_2_key


def save_json_dict(data, filename, mode='w', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)


def pad_list(item_list, pad, max_len):
    item_list = item_list[:max_len]
    return item_list + [pad] * (max_len - len(item_list))


def pad_batch(data_batch, pad, max_len=None):
    if max_len is None:
        max_len = len(max(data_batch, key=len))
    return [pad_list(data, pad, max_len) for data in data_batch]


def convert_item(item, convert_dict, unk):
    return convert_dict[item] if item in convert_dict else unk


def convert_list(item_list, convert_dict, pad, unk, max_len=None):
    item_list = [convert_item(item, convert_dict, unk) for item in item_list]
    if max_len is not None:
        item_list = pad_list(item_list, pad, max_len)

    return item_list


def make_batch_iter(data, batch_size, shuffle):
    data_size = len(data)
    num_batches = (data_size + batch_size - 1) // batch_size

    if shuffle:
        random.shuffle(data)
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(data_size, (i + 1) * batch_size)
        yield data[start_index:end_index]


def load_glove_embedding(data_file, word_list):
    w2v = {}
    with open(data_file, 'r', encoding='utf-8') as fin:
        line = fin.readline()
        embedding_size = len(line.strip().split()) - 1
        while line and line != '':
            line = line.strip().split()
            if len(line) == embedding_size + 1:
                word = line[0]
                vector = [float(val) for val in line[1:]]
                if word in word_list:
                    w2v[word] = vector
            line = fin.readline()
    logger.info('hit words: {}'.format(len(w2v)))

    embedding_matrix = []
    for word in word_list:
        if word in w2v:
            embedding_matrix.append(w2v[word])
        else:
            embedding_matrix.append([0.0] * embedding_size)
    return np.array(embedding_matrix), embedding_size


# ====================
def generate_outputs(start_predicted, end_predicted, phrase_predicted):
    outputs = []
    for start_flags, end_flags, phrase_flags in zip(start_predicted, end_predicted, phrase_predicted):
        item = {
            'predicted_start': [i for i, is_start in enumerate(start_flags) if is_start],
            'predicted_end': [i for i, is_end in enumerate(end_flags) if is_end],
            'predicted': []
        }
        for i in item['predicted_start']:
            for j in item['predicted_end']:
                if phrase_flags[i][j] == 1:
                    item['predicted'].append((i, j))
        outputs.append(item)
    return outputs


def refine_outputs(examples, outputs):
    refined_outputs = []
    for example, entry in zip(examples, outputs):
        context = example.context
        token_spans = example.token_spans
        golden_start = [start for phrase, start, end in example.phrase_spans]
        golden_end = [end for phrase, start, end in example.phrase_spans]
        golden = [
            (context[token_spans[start][1]:token_spans[end][2]], token_spans[start][1], token_spans[end][2])
            for phrase, start, end in example.phrase_spans
        ]
        predicted = [
            (context[token_spans[start][1]:token_spans[end][2]], token_spans[start][1], token_spans[end][2])
            for start, end in entry['predicted']
        ]
        refined_outputs.append({
            'context': context,
            'golden': golden,
            'predicted': predicted,
            'golden_start': golden_start,
            'predicted_start': entry['predicted_start'],
            'golden_end': golden_end,
            'predicted_end': entry['predicted_end'],
        })
    return refined_outputs


def compute_metrics(outputs):
    results = {'Golden': 0, 'Predicted': 0, 'Matched': 0}
    for item in outputs:
        golden = set([v[0] for v in item['golden']])
        predicted = set([v[0] for v in item['predicted']])
        results['Golden'] += len(golden)
        results['Predicted'] += len(predicted)
        results['Matched'] += len(golden & predicted)
    if results['Golden'] == 0:
        if results['Predicted'] == 0:
            results['Precision'] = results['Recall'] = results['F1'] = 1.0
        else:
            results['Precision'] = results['Recall'] = results['F1'] = 0.0
    else:
        if results['Matched'] == 0 or results['Predicted'] == 0:
            results['Precision'] = results['Recall'] = results['F1'] = 0.0
        else:
            results['Precision'] = results['Matched'] / results['Predicted']
            results['Recall'] = results['Matched'] / results['Golden']
            results['F1'] = 2 * results['Precision'] * results['Recall'] / (results['Precision'] + results['Recall'])

    return results
