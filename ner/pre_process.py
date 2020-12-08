# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/23
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2020/10/19
"""

import logging
import stanza
from tqdm import tqdm

from utils import init_logger, read_json_lines

logger = logging.getLogger(__name__)


def convert_data(filein, fileout):
    nlp = stanza.Pipeline('en')
    fp = open(fileout, 'w', encoding='utf-8')
    for line in tqdm(list(read_json_lines(filein)), desc='Converting data'):
        context = line['context']
        doc = nlp(context)
        tokens = []
        index_mapping = {}
        for i, token in enumerate(doc.iter_tokens()):
            tokens.append(token.text)
            for j in range(token.start_char, token.end_char):
                index_mapping[j] = i
            index_mapping[token.end_char] = i + 1
        tags = ['O'] * len(tokens)
        qas = sorted(line['qas'], key=lambda x: (x['answer_start'], -x['answer_end']))
        for qa in qas:
            start = index_mapping[qa['answer_start']]
            end = index_mapping[qa['answer_end']]
            if not all([v == 'O' for v in tags[start:end]]):  # skip overlapping tokens
                continue
            tags[start] = 'B-Answer'
            for i in range(start + 1, end):
                tags[i] = 'I-Answer'
        for token, tag in zip(tokens, tags):
            print('{} {}'.format(token, tag), file=fp)
        print('', file=fp)


def main():
    init_logger(logging.INFO)

    convert_data('../data/phrase/data_train.json', '../data/ner/data_train.txt')
    convert_data('../data/phrase/data_eval.json', '../data/ner/data_eval.txt')
    convert_data('../data/phrase/data_test.json', '../data/ner/data_test.txt')


if __name__ == '__main__':
    main()
