# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/6/16
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2020/6/16
"""

import re
import logging
import tokenizations
from tqdm import tqdm
from collections import defaultdict
from allennlp.predictors.predictor import Predictor

from utils import init_logger, read_file, read_json_lines, save_json_lines

logger = logging.getLogger(__name__)


def extract_phrases(root):
    def _locate_sequence(sequence_a, sequence_b, shift):
        for i in range(shift, len(sequence_a)):
            if sequence_a[i:i + len(sequence_b)] == sequence_b:
                return i, i + len(sequence_b)

    def _dfs(parent, parent_shift):
        phrase_spans = set()
        child_shift = 0
        for child in parent['children']:
            tokens = child['word'].split()
            node_start, node_end = _locate_sequence(parent['word'].split(), tokens, child_shift)
            if 'children' in child:
                phrase_spans.add((node_start + parent_shift, node_end + parent_shift, 0))
                phrase_spans.update(_dfs(child, parent_shift + child_shift))
            child_shift += len(tokens)
        return phrase_spans

    return _dfs(root, 0)


def locate_answer(token_spans, answer_start, answer_end):
    start, end = 0, len(token_spans)
    while start < end - 1 and token_spans[start + 1][0] <= answer_start:
        start += 1
    while start < end - 1 and token_spans[end - 1][0] >= answer_end:
        end -= 1

    return start, end


def convert_data(filein, fileout):
    data = defaultdict(list)
    for line in tqdm(list(read_file(filein)), desc='Converting data: {}'.format(filein)):
        _, _, _, _, _, _, context, answer_start, answer, question = line.strip().split('\t')
        answer = re.sub(r'[,.?]+$', '', answer)
        # answer = re.sub(r'[`~!@#$%^&*()\-_=+\[{\]}\\|;:\'",<.>/?]+$', '', answer)
        answer_start = int(answer_start)
        answer_end = answer_start + len(answer)
        if context[answer_start:answer_end] != answer: continue
        data[context].append({
            'question': question,
            'answer_start': answer_start,
            'answer_end': answer_end,
        })
    data = [{'context': key, 'qas': value} for key, value in data.items()]
    save_json_lines(data, fileout)


def tokenize_data(filein, fileout, cuda_device=-1):
    predictor = Predictor.from_path(
        'https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz',
        cuda_device=cuda_device,
    )

    data = []
    for line in tqdm(list(read_json_lines(filein)), desc='Tokenizing data'):
        context = line['context']
        encoded_context = predictor.predict(context)
        tokenized_context = encoded_context['tokens']
        token_spans = tokenizations.get_original_spans(tokenized_context, context)
        phrase_spans = extract_phrases(encoded_context['hierplane_tree']['root'])
        phrase_spans = sorted(phrase_spans, key=lambda x: (x[1] - x[0], x[0]))

        qas = []
        for qa in line['qas']:
            question = qa['question']
            answer_start = qa['answer_start']
            answer_end = qa['answer_end']
            encoded_question = predictor.predict(question)
            tokenized_question = encoded_question['tokens']

            flag = True
            answer_span = locate_answer(token_spans, answer_start, answer_end)
            for i in range(len(phrase_spans)):
                start, end, label = phrase_spans[i]
                if start <= answer_span[0] < answer_span[1] <= end:
                    if (start, end) == answer_span:
                        # logger.info('Exact match with {}'.format(tokenized_context[start:end]))
                        phrase_spans[i] = (start, end, 1)
                    else:
                        # logger.info('Fuzzy match with {}'.format(tokenized_context[start:end]))
                        phrase_spans[i] = (start, end, 2)
                    qa = {'question': tokenized_question, 'answer_start': start, 'answer_end': end}
                    if qa not in qas:
                        qas.append(qa)
                    flag = False
                    break
            if flag:
                logger.info('=' * 20)
                logger.info(context)
                logger.info(context[answer_start:answer_end])
                logger.info(tokenized_context[answer_span[0]:answer_span[1]])

        if len(qas) > 0:
            data.append({
                'context': tokenized_context,
                'phrases': phrase_spans,
                'qas': qas
            })

    save_json_lines(data, fileout)


def count(filename):
    mismatch = 0
    exact_match = 0
    fuzzy_match = 0
    file_iter = tqdm(list(read_json_lines(filename)), desc='Counting')
    for line in file_iter:
        for phrase in line['phrases']:
            if phrase[-1] == 0: mismatch += 1
            elif phrase[-1] == 1: exact_match += 1
            elif phrase[-1] == 2: fuzzy_match += 1
            else: raise ValueError(phrase)
    logger.info('Mismatch: {}'.format(mismatch))
    logger.info('Exact match: {}'.format(exact_match))
    logger.info('Fuzzy match: {}'.format(fuzzy_match))
    logger.info('Positive rate: {}'.format((exact_match + fuzzy_match) / (mismatch + exact_match + fuzzy_match)))


def main():
    init_logger(logging.INFO)

    # convert_data('data/redistribute/raw/train.txt', 'data/phrase/train.json')
    # convert_data('data/redistribute/raw/dev.txt.shuffle.dev', 'data/phrase/dev.json')
    # convert_data('data/redistribute/raw/dev.txt.shuffle.test', 'data/phrase/test.json')

    # tokenize_data('data/phrase/train.json', 'data/phrase/train.tokenized.json', cuda_device=0)
    # tokenize_data('data/phrase/dev.json', 'data/phrase/dev.tokenized.json', cuda_device=0)
    # tokenize_data('data/phrase/test.json', 'data/phrase/test.tokenized.json', cuda_device=0)

    count('data/phrase/train.tokenized.json')
    count('data/phrase/dev.tokenized.json')
    count('data/phrase/test.tokenized.json')


if __name__ == '__main__':
    main()
